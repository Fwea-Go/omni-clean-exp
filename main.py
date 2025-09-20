# FWEA-I Clean Editor - Optimized FastAPI Backend
# Enhanced version with better file size handling and error management

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os, io, json, re, tempfile, subprocess, uuid, requests
import asyncio
from pathlib import Path
import time

# -------- ENHANCED CONFIG --------
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
WHISPER_MODEL = "@cf/openai/whisper"

import re
# Aggressive profanity detection regex (catches leetspeak/obfuscated variants)
PROFANITY_RE = re.compile(
    r'\b(?:f[\W_]*u[\W_]*c[\W_]*k+|s[\W_]*h[\W_]*i[\W_]*t+|b[\W_]*i[\W_]*t[\W_]*c[\W_]*h+|n[\W_]*i[\W_]*g+[\W_]*a+)\b',
    re.I
)

# Normalize tokens to catch leetspeak/obfuscations (e.g., f*ck, f@ck, f#ck)
LEET_MAP = str.maketrans({
    '@':'a','$':'s','0':'o','1':'i','!':'i','3':'e','4':'a','5':'s','7':'t'
})
def norm_token(s: str) -> str:
    s = (s or '').lower()
    s = s.translate(LEET_MAP)
    s = re.sub(r'[^a-z]+', '', s)  # strip non-letters
    return s

# pad around detected words to avoid clicks, and merge close gaps
MUTE_PAD = 0.15   # seconds before/after
MERGE_GAP = 0.12  # merge spans separated by <= this

# File size limits
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CHUNK_SIZE = 5 * 1024 * 1024    # 5MB for Whisper API

app = FastAPI(
    title="FWEA-I Clean Editor API",
    description="Optimized audio processing with enhanced file size handling",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve generated previews
os.makedirs("public", exist_ok=True)
app.mount("/public", StaticFiles(directory="public"), name="public")

def run_command(cmd, timeout=30):
    """Enhanced command runner with timeout"""
    try:
        p = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=timeout
        )
        if p.returncode != 0:
            raise RuntimeError(p.stderr)
        return p.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timeout after {timeout}s")

def get_file_size(filepath: str) -> int:
    """Get file size safely"""
    try:
        return os.path.getsize(filepath)
    except:
        return 0

def ffprobe_duration(src_path: str) -> float:
    """Enhanced duration detection with fallback"""
    try:
        out = run_command([
            "ffprobe", "-v", "quiet", 
            "-show_entries", "format=duration", 
            "-of", "csv=p=0", src_path
        ], timeout=10)
        duration = float(out.strip())
        return duration if duration > 0 else 0.0
    except Exception as e:
        print(f"Duration probe failed: {e}")
        return 0.0

def make_optimized_slice(src_path: str, start: float, length: float, level: int = 1) -> str:
    """Create optimized audio slice with multiple compression levels"""
    
    # Optimization levels - progressively more aggressive
    configs = [
        {"hz": 32000, "br": "64k", "codec": "libmp3lame", "ext": "mp3"},   # Level 1: Standard
        {"hz": 24000, "br": "48k", "codec": "libmp3lame", "ext": "mp3"},   # Level 2: Reduced
        {"hz": 16000, "br": "32k", "codec": "libmp3lame", "ext": "mp3"},   # Level 3: Minimal
        {"hz": 16000, "br": "24k", "codec": "libopus", "ext": "ogg"},      # Level 4: Opus
        {"hz": 8000, "br": "16k", "codec": "libopus", "ext": "ogg"},       # Level 5: Ultra minimal
    ]
    
    config = configs[min(level - 1, len(configs) - 1)]
    
    fd, tmpaud = tempfile.mkstemp(suffix=f".{config['ext']}")
    os.close(fd)
    
    # Adjust duration for higher levels
    if level > 2:
        length = min(length, 1.5)  # Max 1.5s for levels 3+
    if level > 4:
        length = min(length, 1.0)  # Max 1s for level 5
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src_path,
        "-ss", f"{max(0.0, start)}",
        "-t", f"{max(0.1, length)}",
        "-ac", "1",  # Force mono
        "-ar", str(config["hz"]),
        "-c:a", config["codec"],
    ]
    
    if config["codec"] == "libopus":
        cmd.extend(["-b:a", config["br"], "-vbr", "on", "-compression_level", "10"])
    else:
        cmd.extend(["-b:a", config["br"], "-q:a", "9"])  # Highest compression
    
    cmd.append(tmpaud)
    
    try:
        run_command(cmd, timeout=15)
        
        # Verify output
        size = get_file_size(tmpaud)
        if size < 4096:
            os.unlink(tmpaud)
            raise RuntimeError(f"Slice too small: {size} bytes")
        
        print(f"[slice L{level}] {os.path.basename(tmpaud)} len={length:.2f}s size={size/1024:.1f}KB codec={config['codec']}")
        return tmpaud
        
    except Exception as e:
        if os.path.exists(tmpaud):
            os.unlink(tmpaud)
        raise e

def whisper_file_enhanced(filepath: str, level: int = 1) -> dict:
    """Enhanced Whisper API call with better error handling"""
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("Missing CF_ACCOUNT_ID / CF_API_TOKEN")
    
    file_size = get_file_size(filepath)
    
    # Pre-flight size check
    if file_size > MAX_CHUNK_SIZE:
        raise RuntimeError(f"File too large for Whisper API: {file_size/1024/1024:.1f}MB")
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{WHISPER_MODEL}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    data = {"input": json.dumps({"response_format": "verbose_json"})}
    
    print(f"[whisper L{level}] {os.path.basename(filepath)} bytes={file_size}")
    
    try:
        with open(filepath, "rb") as f:
            mime = "audio/ogg" if filepath.lower().endswith(".ogg") else "audio/mpeg"
            files = {"file": (os.path.basename(filepath), f, mime)}
            r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        
        if not r.ok:
            error_text = r.text.lower()
            is_too_large = any(phrase in error_text for phrase in [
                "too large", "entity too large", "payload too large", "request too large"
            ])
            is_rate_limit = r.status_code == 429 or "rate limit" in error_text
            
            return {
                "success": False,
                "error": {
                    "status": r.status_code,
                    "message": r.text,
                    "file": os.path.basename(filepath),
                    "bytes": file_size,
                    "level": level
                },
                "is_too_large": is_too_large,
                "is_rate_limit": is_rate_limit,
                "should_retry": is_too_large or is_rate_limit
            }
        
        return {"success": True, "data": r.json()}
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": str(e),
                "file": os.path.basename(filepath),
                "bytes": file_size,
                "level": level
            },
            "is_too_large": False,
            "is_rate_limit": False,
            "should_retry": False
        }

@app.get("/")
async def root():
    return {
        "ok": True,
        "message": "FWEA-I Clean Editor API (Optimized)",
        "version": "2.0",
        "limits": {
            "max_file_size": f"{MAX_UPLOAD_SIZE//1024//1024}MB",
            "max_chunk_duration": "2s",
            "supported_formats": ["mp3", "wav", "m4a", "aac", "flac", "ogg"]
        }
    }

@app.get("/health")
async def health():
    return {
        "ok": True,
        "timestamp": time.time(),
        "environment": {
            "has_cloudflare_credentials": bool(CF_ACCOUNT_ID and CF_API_TOKEN),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "fastapi_version": "Latest"
        }
    }

# Version marker endpoint for quick sanity checks
@app.get("/__version")
async def version():
    return {
        "ok": True,
        "backend": "fastapi",
        "name": "FWEA-I Clean Editor API (Optimized)",
        "version": "2.0",
        "build": "ladder-1 mp3/opus chunks",
        "time": time.time()
    }

@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    start_time = time.time()
    temp_files = []
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file
        suffix = Path(file.filename).suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Check file size during upload
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            tmp.close()
            os.unlink(tmp.name)
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "file_too_large",
                    "message": f"File exceeds {MAX_UPLOAD_SIZE//1024//1024}MB limit",
                    "file_size": len(content),
                    "max_size": MAX_UPLOAD_SIZE
                }
            )
        
        tmp.write(content)
        tmp.flush()
        tmp.close()
        src = tmp.name
        
        print(f"[upload] {file.filename} size={len(content)/1024/1024:.1f}MB")
        
        # Get audio duration
        dur = ffprobe_duration(src)
        if dur == 0:
            os.unlink(src)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_audio_file",
                    "message": "Could not read audio file. Please check format."
                }
            )
        
        print(f"[audio] duration={dur:.1f}s")
        
        # Process in smaller chunks
        INITIAL_CHUNK_SEC = 2.0  # Start with 2s chunks
        full_segments = []
        word_spans = []  # word-level profanity hit windows
        language = "auto"
        
        retry_count = 0
        max_retries = 3
        
        t = 0.0
        while t < dur:
            chunk_length = min(INITIAL_CHUNK_SEC, dur - t)
            success = False
            
            # Try multiple optimization levels
            for level in range(1, 6):
                try:
                    # Create optimized slice
                    slice_path = make_optimized_slice(src, t, chunk_length, level)
                    temp_files.append(slice_path)
                    
                    # Call Whisper API with robust error handling
                    import aiohttp
                    WHISPER_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{WHISPER_MODEL}"
                    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
                    data = {"input": json.dumps({"response_format": "verbose_json"})}
                    try:
                        async with aiohttp.ClientSession() as session:
                            with open(slice_path, "rb") as f:
                                mime = "audio/ogg" if slice_path.lower().endswith(".ogg") else "audio/mpeg"
                                form = aiohttp.FormData()
                                form.add_field("input", json.dumps({"response_format": "verbose_json"}))
                                form.add_field("file", f, filename=os.path.basename(slice_path), content_type=mime)
                                try:
                                    async with session.post(WHISPER_URL, data=form, headers=headers) as resp:
                                        text = await resp.text()
                                        try:
                                            out = json.loads(text)
                                        except Exception:
                                            return web.json_response({"error": "Whisper parse failed", "detail": text}, status=502)
                                except Exception as e:
                                    return web.json_response({"error": "Whisper request failed", "detail": str(e)}, status=502)
                    except Exception as e:
                        return web.json_response({"error": "Whisper request failed", "detail": str(e)}, status=502)
                    # If we got here, out is parsed JSON
                    data = out.get("result", out)
                    if data.get("language"):
                        language = data["language"]
                    segments = data.get("segments", [])
                    for s in segments:
                        st = float(s.get("start", 0.0)) + t
                        en = float(s.get("end", 0.0)) + t
                        txt = s.get("text", "") or ""
                        full_segments.append({
                            "start": st,
                            "end": en,
                            "text": txt
                        })
                        # If word timestamps are present, inspect each word for profanity
                        words = s.get("words") or []
                        for w in words:
                            wtxt = str(w.get("word","")).strip()
                            if not wtxt:
                                continue
                            nrm = norm_token(wtxt)
                            if nrm and PROFANITY_RE.search(wtxt) or PROFANITY_RE.search(nrm):
                                wst = float(w.get("start", st)) + t
                                wen = float(w.get("end", en)) + t
                                word_spans.append({
                                    "start": max(0.0, wst - MUTE_PAD),
                                    "end": max(0.0, wen + MUTE_PAD),
                                    "reason": "profanity"
                                })
                    success = True
                    print(f"[success] chunk {t:.1f}s processed with level {level}")
                    break
                except Exception as e:
                    print(f"[error] level {level}: {e}")
                    if level == 5:
                        return JSONResponse(
                            status_code=502,
                            content={
                                "error": "whisper_failed",
                                "detail": str(e),
                                "chunk": {"start": t, "length": chunk_length, "level": level},
                                "suggestion": "Try a shorter audio file or reduce quality"
                            }
                        )
            
            t += INITIAL_CHUNK_SEC
        
        # Build profanity spans with word-level precision when available
        transcript_text = " ".join(s.get("text", "").strip() for s in full_segments).strip()

        # Prefer word-level hits; otherwise fall back to segment-level text scan
        spans = list(word_spans)
        if not spans:
            for s in full_segments:
                txt = s.get("text", "")
                nrm = norm_token(txt)
                if PROFANITY_RE.search(txt) or PROFANITY_RE.search(nrm):
                    spans.append({
                        "start": max(0.0, s["start"] - MUTE_PAD),
                        "end": max(0.0, s["end"] + MUTE_PAD),
                        "reason": "profanity"
                    })

        spans.sort(key=lambda x: x["start"])

        # Merge overlapping/nearby spans
        merged = []
        for s in spans:
            if not merged:
                merged.append(dict(s))
                continue
            if s["start"] <= merged[-1]["end"] + MERGE_GAP:
                merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            else:
                merged.append(dict(s))
        
        # Generate preview
        PREVIEW_T = 30.0
        to_mute = []
        
        for s in merged:
            if s["end"] > 0 and s["start"] < PREVIEW_T:
                to_mute.append({
                    "start": max(0.0, s["start"]),
                    "end": min(PREVIEW_T, s["end"])
                })
        
        # Build audio filter
        filters = []
        for s in to_mute:
            if s["end"] > s["start"]:
                filters.append(f"volume=enable='between(t,{s['start']:.3f},{s['end']:.3f})':volume=0")
        
        mute_filter = ",".join(filters) if filters else None
        
        # Generate preview file
        out_id = uuid.uuid4().hex
        out_path = os.path.join("public", f"{out_id}.mp3")
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src, "-t", str(int(PREVIEW_T)),
            "-ac", "2",  # Stereo for preview
            "-ar", "44100",  # Standard quality
            "-codec:a", "libmp3lame", "-b:a", "192k"
        ]
        
        if mute_filter:
            cmd.insert(-4, "-af")
            cmd.insert(-4, mute_filter)
        
        cmd.append(out_path)
        
        try:
            run_command(cmd, timeout=30)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "preview_generation_failed", "detail": str(e)}
            )
        
        processing_time = time.time() - start_time
        
        print(f"[complete] processed in {processing_time:.1f}s, {len(full_segments)} segments, {len(merged)} muted spans")
        
        return {
            "preview_url": f"/public/{os.path.basename(out_path)}",
            "language": language,
            "transcript": transcript_text,
            "muted_spans": merged,
            "metadata": {
                "processing_time_seconds": processing_time,
                "segment_count": len(full_segments),
                "original_duration": dur,
                "original_size": len(content),
                "muted_span_count": len(merged)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_failed",
                "detail": str(e),
                "suggestion": "Please try with a smaller or different audio file"
            }
        )
    finally:
        # Cleanup
        try:
            if 'src' in locals() and os.path.exists(src):
                os.unlink(src)
        except:
            pass
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

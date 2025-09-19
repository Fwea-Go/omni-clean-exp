# FWEA-I Professional Audio Cleaning API - FastAPI Version
# Production-ready with enhanced cleaning and higher limits

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os, io, json, re, tempfile, subprocess, uuid, requests
import asyncio
from pathlib import Path
import time
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PRODUCTION CONFIG
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
WHISPER_MODEL = "@cf/openai/whisper"

# INCREASED LIMITS for production
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
MAX_CHUNK_SIZE = 8 * 1024 * 1024     # 8MB for Whisper API
CHUNK_DURATION = 10.0                # 10-second chunks for better processing

# COMPREHENSIVE PROFANITY PATTERNS - Enhanced multilingual detection
BAD_WORD_PATTERNS = [
    # English (comprehensive)
    r"\b(fuck|fucking|fucked|fucker|shit|shitting|shitted|bitch|bitches|asshole|assholes|cunt|cunts|motherfucker|motherfuckers|dick|dicks|pussy|pussies|cock|cocks|nigga|niggas|nigger|niggers|faggot|faggots|whore|whores|slut|sluts|damn|damned|hell|bastard|bastards|piss|pissed|tits|boobs|boob|ass|asses)\b",
    
    # Spanish
    r"\b(puta|putas|puto|putos|pendejo|pendejos|mierda|mierdas|cabron|cabrones|coño|coños|joder|jodido|carajo|carajos|culo|culos|perra|perras|marica|maricas|maricón|maricones|hijo\s+de\s+puta|hijos\s+de\s+puta)\b",
    
    # French
    r"\b(putain|putains|merde|merdes|salope|salopes|con|cons|connard|connards|connasse|connasses|enculé|enculés|fils\s+de\s+pute|bâtard|bâtards|salopard|salopards)\b",
    
    # Portuguese
    r"\b(merda|merdas|caralho|caralhos|porra|porras|puta|putas|filho\s+da\s+puta|filhos\s+da\s+puta|buceta|bucetas|cu|cus|fdp|desgraça|desgraças)\b",
    
    # Italian
    r"\b(merda|merdate|cazzo|cazzi|puttana|puttane|stronzo|stronzi|figlio\s+di\s+puttana|figli\s+di\s+puttana|vaffanculo|porco\s+dio|madonna\s+mia)\b",
    
    # German
    r"\b(scheiße|scheisse|fick|ficken|arschloch|arschlöcher|fotze|fotzen|hurensohn|hurensöhne|verdammt|scheiss|dumme\s+sau|blöde\s+kuh)\b",
    
    # Creative spellings and censored versions
    r"\b(f\*ck|f\*\*k|sh\*t|b\*tch|a\*\*hole|d\*mn|h\*ll|fuk|shyt|beyotch|azz|dayum)\b"
]

BAD_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in BAD_WORD_PATTERNS]

app = FastAPI(
    title="FWEA-I Professional Audio Cleaning API",
    description="Production-ready audio processing with enhanced profanity detection and professional cleaning",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Setup directories
for directory in ["public", "uploads", "temp"]:
    os.makedirs(directory, exist_ok=True)

app.mount("/public", StaticFiles(directory="public"), name="public")

def run_command(cmd: str, timeout: int = 60) -> str:
    """Enhanced command runner with comprehensive error handling"""
    logger.info(f"[EXEC] {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=timeout,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"[EXEC_ERROR] Command failed: {result.stderr}")
            raise RuntimeError(f"Command failed: {result.stderr}")
        
        return result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timeout after {timeout}s")
    except Exception as e:
        logger.error(f"[EXEC_EXCEPTION] {str(e)}")
        raise RuntimeError(f"Command execution failed: {str(e)}")

def get_file_stats(filepath: str) -> Dict:
    """Get comprehensive file statistics"""
    try:
        stat = os.stat(filepath)
        return {
            "size": stat.st_size,
            "exists": True,
            "readable": os.access(filepath, os.R_OK)
        }
    except:
        return {"size": 0, "exists": False, "readable": False}

def detect_profanity(text: str) -> Tuple[bool, int]:
    """Enhanced profanity detection with word count"""
    if not text or not isinstance(text, str):
        return False, 0
    
    total_matches = 0
    has_profanity = False
    
    for regex in BAD_REGEX:
        matches = regex.findall(text)
        if matches:
            has_profanity = True
            total_matches += len(matches)
    
    return has_profanity, total_matches

def get_audio_duration(filepath: str) -> float:
    """Get audio duration with multiple fallback methods"""
    methods = [
        f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{filepath}"',
        f'ffprobe -v quiet -select_streams a:0 -show_entries stream=duration -of csv=p=0 "{filepath}"',
    ]
    
    for method in methods:
        try:
            output = run_command(method, timeout=15)
            duration = float(output.strip())
            if duration > 0:
                logger.info(f"[DURATION] {duration:.2f}s detected")
                return duration
        except Exception as e:
            logger.warning(f"[DURATION] Method failed: {e}")
    
    return 0.0

def create_audio_slice(src_path: str, start: float, duration: float, level: int = 1) -> str:
    """Create optimized audio slice with progressive compression levels"""
    
    # Progressive optimization configurations
    configs = [
        {"hz": 44100, "br": "128k", "codec": "libmp3lame", "ext": "mp3", "ac": 2},  # High quality
        {"hz": 32000, "br": "96k", "codec": "libmp3lame", "ext": "mp3", "ac": 2},   # Medium quality
        {"hz": 24000, "br": "64k", "codec": "libmp3lame", "ext": "mp3", "ac": 1},   # Low quality
        {"hz": 16000, "br": "48k", "codec": "libmp3lame", "ext": "mp3", "ac": 1},   # Minimal
        {"hz": 16000, "br": "32k", "codec": "libopus", "ext": "ogg", "ac": 1},      # Opus low
        {"hz": 8000, "br": "24k", "codec": "libopus", "ext": "ogg", "ac": 1},       # Opus minimal
    ]
    
    config = configs[min(level - 1, len(configs) - 1)]
    
    # Adjust duration for higher compression levels
    if level > 3:
        duration = min(duration, 4.0)  # Max 4s for levels 4+
    if level > 5:
        duration = min(duration, 2.0)  # Max 2s for level 6
    
    # Generate unique filename
    timestamp = int(time.time() * 1000) % 1000000
    filename = f"slice_{timestamp}_L{level}.{config['ext']}"
    output_path = os.path.join("temp", filename)
    
    # Build FFmpeg command
    cmd_parts = [
        "ffmpeg -hide_banner -loglevel warning -y",
        f'-i "{src_path}"',
        f"-ss {start:.3f}",
        f"-t {duration:.3f}",
        f"-ac {config['ac']}",
        f"-ar {config['hz']}",
        f"-c:a {config['codec']}"
    ]
    
    if config["codec"] == "libopus":
        cmd_parts.extend(["-b:a", config["br"], "-vbr on", "-compression_level 10", "-application voip"])
    else:
        cmd_parts.extend(["-b:a", config["br"], "-q:a", "9" if level > 3 else "7"])
    
    cmd_parts.append(f'"{output_path}"')
    
    try:
        run_command(" ".join(cmd_parts), timeout=30)
        
        stats = get_file_stats(output_path)
        if not stats["exists"] or stats["size"] < 1024:
            raise RuntimeError(f"Output file too small: {stats['size']} bytes")
        
        logger.info(f"[SLICE] Created {filename} - {stats['size']/1024:.1f}KB (level {level})")
        return output_path
        
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(f"Slice creation failed (level {level}): {str(e)}")

async def transcribe_audio(filepath: str, metadata: Dict = {}) -> Dict:
    """Enhanced Whisper transcription with retry logic"""
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError("Missing Cloudflare credentials")
    
    stats = get_file_stats(filepath)
    if stats["size"] > MAX_CHUNK_SIZE:
        raise RuntimeError(f"File too large: {stats['size']/1024/1024:.1f}MB > {MAX_CHUNK_SIZE/1024/1024}MB")
    
    logger.info(f"[WHISPER] Processing {os.path.basename(filepath)} ({stats['size']/1024:.1f}KB)")
    
    # Prepare request
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{WHISPER_MODEL}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    
    with open(filepath, "rb") as f:
        files = {"file": (os.path.basename(filepath), f, "audio/mpeg")}
        data = {"input": json.dumps({"response_format": "verbose_json"})}
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, files=files, data=data, timeout=90)
                
                if response.ok:
                    result = response.json()
                    logger.info(f"[WHISPER] Success - {os.path.basename(filepath)}")
                    return {"success": True, "data": result}
                
                # Handle errors
                error_text = response.text.lower()
                is_rate_limit = response.status_code == 429 or "rate limit" in error_text
                is_too_large = any(phrase in error_text for phrase in ["too large", "entity too large", "payload too large"])
                
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff
                    logger.warning(f"[WHISPER] Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                return {
                    "success": False,
                    "error": {
                        "status": response.status_code,
                        "message": response.text,
                        "is_too_large": is_too_large,
                        "is_rate_limit": is_rate_limit
                    }
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": {
                            "message": str(e),
                            "attempt": attempt + 1
                        }
                    }
                await asyncio.sleep(1)

def generate_professional_clean_audio(src_path: str, muted_spans: List[Dict], output_path: str, duration: float = 30.0) -> None:
    """Generate professionally cleaned audio with smooth crossfades"""
    
    if not muted_spans:
        # No profanity - create high-quality preview
        cmd = f'''ffmpeg -hide_banner -loglevel warning -y -i "{src_path}" -t {duration} -c:a libmp3lame -b:a 320k -ar 44100 -ac 2 "{output_path}"'''
        run_command(cmd, timeout=60)
        return
    
    # Process spans for preview duration
    preview_spans = []
    for span in muted_spans:
        if span["start"] < duration and span["end"] > 0:
            preview_spans.append({
                "start": max(0, span["start"]),
                "end": min(duration, span["end"])
            })
    
    # Sort and merge overlapping spans
    preview_spans.sort(key=lambda x: x["start"])
    merged_spans = []
    
    for span in preview_spans:
        if not merged_spans or span["start"] > merged_spans[-1]["end"]:
            merged_spans.append(span)
        else:
            merged_spans[-1]["end"] = max(merged_spans[-1]["end"], span["end"])
    
    if not merged_spans:
        # No spans in preview range
        generate_professional_clean_audio(src_path, [], output_path, duration)
        return
    
    logger.info(f"[CLEAN] Processing {len(merged_spans)} profanity spans with professional crossfades")
    
    # Create sophisticated volume filter with smooth crossfades
    CROSSFADE_MS = 100  # 100ms crossfade for ultra-smooth transitions
    crossfade_duration = CROSSFADE_MS / 1000.0
    
    volume_filters = []
    for span in merged_spans:
        start_time = span["start"]
        end_time = span["end"]
        
        # Create smooth fade out before mute
        fade_out_start = max(0, start_time - crossfade_duration)
        fade_out_end = start_time
        
        # Create smooth fade in after mute
        fade_in_start = end_time
        fade_in_end = min(duration, end_time + crossfade_duration)
        
        # Fade out filter
        if fade_out_start < fade_out_end:
            volume_filters.append(
                f"volume=enable='between(t,{fade_out_start:.3f},{fade_out_end:.3f})':volume='1-((t-{fade_out_start:.3f})/{crossfade_duration:.3f})'"
            )
        
        # Mute filter
        volume_filters.append(
            f"volume=enable='between(t,{start_time:.3f},{end_time:.3f})':volume=0"
        )
        
        # Fade in filter
        if fade_in_start < fade_in_end:
            volume_filters.append(
                f"volume=enable='between(t,{fade_in_start:.3f},{fade_in_end:.3f})':volume='(t-{fade_in_start:.3f})/{crossfade_duration:.3f}'"
            )
    
    # Combine all filters
    audio_filter = ",".join(volume_filters)
    
    # Generate high-quality cleaned audio
    cmd = f'''ffmpeg -hide_banner -loglevel warning -y -i "{src_path}" -t {duration} -af "{audio_filter}" -c:a libmp3lame -b:a 320k -ar 44100 -ac 2 "{output_path}"'''
    
    run_command(cmd, timeout=90)
    
    stats = get_file_stats(output_path)
    logger.info(f"[CLEAN] Professional preview generated: {stats['size']/1024:.1f}KB with {len(merged_spans)} cleaned sections")

@app.get("/")
async def root():
    return {
        "ok": True,
        "message": "FWEA-I Professional Audio Cleaning API",
        "version": "3.0",
        "features": [
            "Professional audio cleaning with smooth crossfades",
            "100MB file upload support",
            "Enhanced multilingual profanity detection",
            "High-quality 320kbps output",
            "Robust chunked processing",
            "Advanced retry mechanisms"
        ],
        "limits": {
            "max_file_size": "100MB",
            "max_processing_time": "5 minutes",
            "chunk_duration": f"{CHUNK_DURATION}s",
            "supported_formats": ["mp3", "wav", "m4a", "aac", "flac", "ogg", "wma"]
        }
    }

@app.get("/health")
async def health():
    return {
        "ok": True,
        "timestamp": time.time(),
        "environment": {
            "has_cloudflare_credentials": bool(CF_ACCOUNT_ID and CF_API_TOKEN),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "directories_ready": {
                "public": os.path.exists("public"),
                "uploads": os.path.exists("uploads"), 
                "temp": os.path.exists("temp")
            }
        }
    }

@app.post("/preview")
async def create_preview(file: UploadFile = File(...)):
    start_time = time.time()
    temp_files = []
    original_file = None
    
    try:
        # Validate file upload
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read and save file
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "file_too_large",
                    "message": f"File exceeds {MAX_UPLOAD_SIZE//1024//1024}MB limit",
                    "file_size": len(content),
                    "max_size": MAX_UPLOAD_SIZE
                }
            )
        
        # Save to temporary file
        suffix = Path(file.filename).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            original_file = tmp.name
        
        logger.info(f"[UPLOAD] {file.filename} - {len(content)/1024/1024:.2f}MB")
        
        # Get audio duration
        duration = get_audio_duration(original_file)
        if duration <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not determine audio duration or invalid audio file"
            )
        
        logger.info(f"[AUDIO] Duration: {duration:.2f}s")
        
        # CHUNKED PROCESSING - Process in segments for better handling
        all_segments = []
        detected_language = "auto"
        total_chunks = int((duration + CHUNK_DURATION - 1) // CHUNK_DURATION)
        
        logger.info(f"[PROCESS] Starting chunked processing ({total_chunks} chunks of {CHUNK_DURATION}s each)")
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * CHUNK_DURATION
            chunk_duration = min(CHUNK_DURATION, duration - chunk_start)
            processed = False
            
            logger.info(f"[CHUNK] Processing chunk {chunk_idx + 1}/{total_chunks}: {chunk_start:.1f}s-{chunk_start + chunk_duration:.1f}s")
            
            # Try progressive optimization levels
            for level in range(1, 7):
                if processed:
                    break
                    
                try:
                    # Create audio slice
                    slice_path = create_audio_slice(original_file, chunk_start, chunk_duration, level)
                    temp_files.append(slice_path)
                    
                    # Transcribe with Whisper
                    result = await transcribe_audio(slice_path, {"level": level, "chunk": chunk_idx})
                    
                    if result["success"]:
                        data = result["data"]
                        result_data = data.get("result", data)
                        
                        if result_data.get("language"):
                            detected_language = result_data["language"]
                        
                        # Process segments
                        segments = result_data.get("segments", [])
                        for segment in segments:
                            all_segments.append({
                                "start": float(segment.get("start", 0)) + chunk_start,
                                "end": float(segment.get("end", 0)) + chunk_start,
                                "text": segment.get("text", "")
                            })
                        
                        processed = True
                        logger.info(f"[SUCCESS] Chunk {chunk_idx + 1} processed with level {level} - {len(segments)} segments")
                        
                    else:
                        error = result["error"]
                        if error.get("is_rate_limit"):
                            logger.warning("[RATE_LIMIT] Waiting before retry...")
                            await asyncio.sleep(2)
                        if error.get("is_too_large") and level < 6:
                            logger.warning(f"[TOO_LARGE] Trying higher compression (level {level + 1})")
                            continue
                        
                        if level == 6:
                            logger.error(f"[CHUNK_FAILED] All levels failed for chunk {chunk_idx + 1}")
                            processed = True  # Skip this chunk
                        
                except Exception as e:
                    logger.error(f"[CHUNK_ERROR] Level {level}: {str(e)}")
                    if level == 6:
                        logger.error(f"[CHUNK_SKIP] Skipping chunk {chunk_idx + 1}")
                        processed = True
        
        logger.info(f"[SEGMENTS] Extracted {len(all_segments)} total segments")
        
        # ENHANCED PROFANITY DETECTION
        profanity_spans = []
        total_profane_words = 0
        
        for segment in all_segments:
            text = segment.get("text", "")
            has_profanity, word_count = detect_profanity(text)
            
            if has_profanity:
                profanity_spans.append({
                    "start": max(0.0, segment["start"]),
                    "end": max(0.0, segment["end"]),
                    "reason": "profanity",
                    "sample_text": text[:50] + "..." if len(text) > 50 else text,
                    "word_count": word_count
                })
                total_profane_words += word_count
        
        logger.info(f"[PROFANITY] Detected {len(profanity_spans)} spans with {total_profane_words} profane words")
        
        # MERGE OVERLAPPING SPANS
        profanity_spans.sort(key=lambda x: x["start"])
        merged_spans = []
        
        for span in profanity_spans:
            if not merged_spans or span["start"] > merged_spans[-1]["end"] + 0.1:
                merged_spans.append({
                    "start": span["start"],
                    "end": span["end"],
                    "reason": span["reason"]
                })
            else:
                merged_spans[-1]["end"] = max(merged_spans[-1]["end"], span["end"])
        
        # GENERATE PROFESSIONAL CLEAN PREVIEW
        preview_id = f"clean_{int(time.time())}{os.urandom(4).hex()}"
        preview_path = os.path.join("public", f"{preview_id}.mp3")
        
        generate_professional_clean_audio(original_file, merged_spans, preview_path, 30.0)
        
        # COMPILE RESULTS
        processing_time = time.time() - start_time
        transcript = " ".join(segment.get("text", "").strip() for segment in all_segments).strip()
        
        # Calculate clean percentage
        muted_duration = sum(span["end"] - span["start"] for span in merged_spans)
        clean_percentage = 100 - (muted_duration / min(duration, 30) * 100) if merged_spans else 100
        
        logger.info(f"[COMPLETE] Processed in {processing_time:.1f}s - {len(merged_spans)} muted spans ({clean_percentage:.1f}% clean)")
        
        return {
            "preview_url": f"/public/{os.path.basename(preview_path)}",
            "language": detected_language,
            "transcript": transcript,
            "muted_spans": merged_spans,
            "metadata": {
                "processing_time_seconds": processing_time,
                "segment_count": len(all_segments),
                "original_duration": duration,
                "original_size": len(content),
                "muted_span_count": len(merged_spans),
                "profane_words_detected": total_profane_words,
                "clean_percentage": round(clean_percentage, 1),
                "chunks_processed": total_chunks,
                "preview_quality": "320kbps Professional"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_failed",
                "message": str(e),
                "suggestion": "Try a shorter audio file or check server configuration"
            }
        )
    finally:
        # CLEANUP
        cleanup_files = [original_file] + temp_files
        for file_path in cleanup_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    logger.info(f"[CLEANUP] Removed {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"[CLEANUP] Could not remove {file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

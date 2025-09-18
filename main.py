from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os, io, json, re, tempfile, subprocess, uuid, requests

# -------- CONFIG --------
# Set these as environment variables (export in shell or systemd)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN  = os.getenv("CF_API_TOKEN", "")
# Cloudflare Whisper model id
WHISPER_MODEL = "@cf/openai/whisper"

# Basic multilingual profanity list (starter; expand as needed)
BAD_WORDS = [
    # English
    r"\b(fuck|shit|bitch|asshole|cunt|motherfucker|dick|pussy|nigga|nigger|hoe|slut)\b",
    # Spanish
    r"\b(puta|puto|pendejo|mierda|cabron|coño)\b",
    # French
    r"\b(putain|merde|salope|con)\b",
    # Haitian Creole (starter)
    r"\b(bitch|koko|vagin|kaka|manmanw|manman’w)\b",
    # Portuguese
    r"\b(merda|caralho|porra|puta)\b",
    # …add more langs/variants
]
BAD_RE = re.compile("|".join(BAD_WORDS), flags=re.IGNORECASE)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

# Serve generated previews
os.makedirs("public", exist_ok=True)
app.mount("/public", StaticFiles(directory="public"), name="public")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p.stdout

# --- FFmpeg/ffprobe helpers ---
def ffprobe_duration(src_path: str) -> float:
    try:
        out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", src_path])
        return float(out.strip())
    except Exception:
        return 0.0

def make_slice(src_path: str, start: float, length: float) -> str:
    fd, tmpaud = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src_path,
        # accurate output trim (put -ss/-t after -i)
        "-ss", f"{max(0.0, start)}",
        "-t", f"{max(0.1, length)}",
        # ultra‑small, CF‑friendly slice: MP3 mono 32 kHz @ 64k
        "-ac", "1",
        "-ar", "32000",
        "-c:a", "libmp3lame",
        "-b:a", "64k",
        tmpaud,
    ]
    run(cmd)
    # sanity: ensure slice is at least ~4KB (prevents empty uploads)
    if os.path.getsize(tmpaud) < 4096:
        raise RuntimeError(f"slice_too_small: {tmpaud}")
    return tmpaud

# Cloudflare Whisper call for a local file path
def whisper_file(filepath: str) -> dict:
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("Missing CF_ACCOUNT_ID / CF_API_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{WHISPER_MODEL}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    # response_format=verbose_json for segments with start/end
    data = {"input": json.dumps({"response_format": "verbose_json"})}
    with open(filepath, "rb") as f:
        files = {"file": (os.path.basename(filepath), f, "audio/mpeg")}
        r = requests.post(url, headers=headers, data=data, files=files, timeout=600)
    if not r.ok:
        raise RuntimeError(f"Cloudflare AI error: {r.status_code} {r.text}")
    return r.json()  # {"result": {"segments":[...], "language":".." }}




@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    # Save upload to temp
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    src = tmp.name

    slices = []
    try:
        # --- Plan chunking ---
        dur = ffprobe_duration(src) or 0.0
        CHUNK_SEC = 5.0
        total = dur if dur > 0 else CHUNK_SEC

        # --- Transcribe per-slice and accumulate segments ---
        full_segments = []
        language = "auto"
        t = 0.0
        while t < total - 1e-6:
            length = CHUNK_SEC if (t + CHUNK_SEC) <= total else (total - t)
            # try default, then progressively smaller/denser if CF complains
            wav = make_slice(src, t, length); slices.append(wav)
            try:
                r = whisper_file(wav)
            except Exception as e:
                # If this smells like a "too large" error, retry with smaller slice/bitrate
                msg = str(e)
                if re.search(r"too\s+large|request.+large", msg, flags=re.I):
                    # shorten to 3s
                    try:
                        wav2 = make_slice(src, t, min(length, 3.0)); slices.append(wav2)
                        r = whisper_file(wav2)
                    except Exception as e2:
                        msg2 = str(e2)
                        if re.search(r"too\s+large|request.+large", msg2, flags=re.I):
                            # shorten further to 2s and drop sample rate/bitrate by re-encoding via ffmpeg args (mp3 path already small)
                            try:
                                # extra-tight encode for last resort: 2s @ 24kHz 48k
                                fd, tmpaud = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
                                cmd = [
                                    "ffmpeg","-hide_banner","-loglevel","error","-y","-i",src,
                                    "-ss",f"{max(0.0,t)}","-t",f"{max(0.1, min(length,2.0))}",
                                    "-ac","1","-ar","24000","-c:a","libmp3lame","-b:a","48k", tmpaud
                                ]
                                run(cmd)
                                slices.append(tmpaud)
                                r = whisper_file(tmpaud)
                            except Exception as e3:
                                return JSONResponse(status_code=502, content={"error":"whisper_failed","detail":str(e3),"chunk":{"start":t,"len":length,"retried":True}})
                        else:
                            return JSONResponse(status_code=502, content={"error":"whisper_failed","detail":msg2,"chunk":{"start":t,"len":length,"retried":True}})
                else:
                    return JSONResponse(status_code=502, content={"error": "whisper_failed", "detail": msg, "chunk": {"start": t, "len": length}})
            result = r.get("result") or r
            if result.get("language"):
                language = result["language"]
            segs = result.get("segments") or []
            for s in segs:
                st = float(s.get("start", 0.0)) + t
                en = float(s.get("end", 0.0)) + t
                full_segments.append({"start": st, "end": en, "text": s.get("text", "")})
            t += CHUNK_SEC

        # --- Profanity spans based on segments ---
        transcript_text = " ".join((s.get("text") or "").strip() for s in full_segments).strip()
        spans = []
        for s in full_segments:
            if BAD_RE.search(s.get("text", "")):
                spans.append({"start": max(0.0, s["start"]), "end": max(0.0, s["end"]), "reason": "profanity"})
        spans.sort(key=lambda x: x["start"]) 
        merged = []
        for s in spans:
            if not merged or s["start"] > merged[-1]["end"]:
                merged.append(dict(s))
            else:
                merged[-1]["end"] = max(merged[-1]["end"], s["end"])

        # --- Render 30s preview from original upload with mutes intersecting [0,30] ---
        PREVIEW_T = 30.0
        to_mute = []
        for s in merged:
            if s["end"] > 0 and s["start"] < PREVIEW_T:
                to_mute.append({
                    "start": max(0.0, s["start"]),
                    "end": min(PREVIEW_T, s["end"]),
                })
        # Build ffmpeg -af filter
        filters = []
        for s in to_mute:
            if s["end"] > s["start"]:
                filters.append(f"volume=enable='between(t,{s['start']:.3f},{s['end']:.3f})':volume=0")
        mute_filter = ",".join(filters) if filters else None

        out_id = uuid.uuid4().hex
        out_path = os.path.join("public", f"{out_id}.mp3")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-t", str(int(PREVIEW_T))]
        if mute_filter:
            cmd += ["-af", mute_filter]
        cmd += ["-codec:a", "libmp3lame", "-b:a", "192k", out_path]
        try:
            run(cmd)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"ffmpeg_failed: {str(e)}"})

        return {
            "preview_url": f"/public/{os.path.basename(out_path)}",
            "language": language,
            "transcript": transcript_text,
            "muted_spans": merged,
        }
    finally:
        # cleanup
        try:
            if os.path.exists(src):
                os.unlink(src)
        except Exception:
            pass
        for p in slices:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

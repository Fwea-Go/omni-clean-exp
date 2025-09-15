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
    r"\\b(fuck|shit|bitch|asshole|cunt|motherfucker|dick|pussy|nigga|nigger|hoe|slut)\\b",
    # Spanish
    r"\\b(puta|puto|pendejo|mierda|cabron|coño)\\b",
    # French
    r"\\b(putain|merde|salope|con)\\b",
    # Haitian Creole (starter)
    r"\\b(bitch|koko|vagin|kaka|manmanw|manman’w)\\b",
    # Portuguese
    r"\\b(merda|caralho|porra|puta)\\b",
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

def whisper_transcribe(file_path):
    # Cloudflare AI /workers-ai endpoint
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("Missing CF_ACCOUNT_ID / CF_API_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{WHISPER_MODEL}"
    with open(file_path, "rb") as f:
        files = {"file": ("audio", f, "application/octet-stream")}
        # You can pass options; word timestamps may not be available – we fall back to segment timing
        data = {"response_format": "verbose_json"}
        headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
        r = requests.post(url, headers=headers, data={"input": json.dumps(data)}, files=files, timeout=600)
    if not r.ok:
        raise RuntimeError(f"Cloudflare AI error: {r.status_code} {r.text}")
    return r.json()  # expected: {"result": {"text": "...", "segments":[{"text": "...","start":0.0,"end":2.4}, ...] }}

def find_profane_spans(transcript_json):
    res = transcript_json.get("result") or transcript_json
    text = res.get("text","") or ""
    segments = res.get("segments") or []  # start/end per chunk
    spans = []
    for seg in segments:
        seg_text = seg.get("text","") or ""
        if BAD_RE.search(seg_text):
            spans.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "reason": "profanity"
            })
    # Merge overlaps
    spans.sort(key=lambda s: s["start"])
    merged = []
    for s in spans:
        if not merged or s["start"] > merged[-1]["end"]:
            merged.append(s)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
    # Clamp negatives, etc.
    for s in merged:
        if s["end"] < s["start"]:
            s["end"] = s["start"]
    # Best-effort language guess (very rough – improve later)
    language = res.get("language") or "auto"
    return text, language, merged

def build_ffmpeg_mute_filter(spans):
    # volume=0 between each span; chain with , to apply sequentially
    # We also lowpass a tiny click-fade via afade at boundaries
    filters = []
    for s in spans:
        start = max(0.0, float(s["start"]))
        end   = max(0.0, float(s["end"]))
        if end <= start: continue
        filters.append(f"volume=enable='between(t,{start:.3f},{end:.3f})':volume=0")
    if not filters:
        return None
    return ",".join(filters)

@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    # Save upload
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    src = tmp.name

    # Transcribe via Cloudflare AI (multilingual)
    try:
        tr = whisper_transcribe(src)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"transcribe_failed: {str(e)}"})

    transcript, language, spans = find_profane_spans(tr)

    # Build FFmpeg filter
    mute_filter = build_ffmpeg_mute_filter(spans)
    out_id = str(uuid.uuid4()).replace("-","")
    out_path = os.path.join("public", f"{out_id}.mp3")

    # FFmpeg command: trim to 30s preview, apply mutes, encode MP3
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", src,
           "-t", "30"]
    if mute_filter:
        cmd += ["-af", mute_filter]
    cmd += ["-codec:a", "libmp3lame", "-b:a", "192k", out_path]
    try:
        run(cmd)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"ffmpeg_failed: {str(e)}"})

    # Done
    return {
        "preview_url": f"/public/{os.path.basename(out_path)}",
        "language": language,
        "transcript": transcript,
        "muted_spans": spans,
    }

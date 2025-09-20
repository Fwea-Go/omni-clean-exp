// backend.js — OPTIMIZED FWEA-I Clean Editor API (Node/Express)
// Enhanced version with better file size handling and error management

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

// Improved promise-wrapped exec with timeout
const sh = (cmd, timeoutMs = 30000) => new Promise((resolve, reject) => {
    const child = exec(cmd, { timeout: timeoutMs }, (e, so, se) => {
        if (e) reject(se || e);
        else resolve(so);
    });
    child.on('timeout', () => reject(new Error('Command timeout')));
});

// Enhanced duration probe with fallback
async function ffprobeDuration(src) {
    try {
        const cmd = `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${src}"`;
        const out = await sh(cmd, 10000);
        const n = parseFloat(String(out).trim());
        return Number.isFinite(n) && n > 0 ? n : 0;
    } catch {
        return 0;
    }
}

// Get file size helper
function getFileSize(filepath) {
    try {
        return fs.statSync(filepath).size;
    } catch {
        return 0;
    }
}

const app = express();
const port = process.env.PORT || 8000;

// Enhanced upload configuration with size limits
const upload = multer({ 
    dest: 'uploads/',
    limits: {
        fileSize: 50 * 1024 * 1024 // 50MB max upload
    }
});

fs.mkdirSync('public', { recursive: true });
fs.mkdirSync('uploads', { recursive: true });

app.use(cors({ origin: '*' }));
app.use('/public', express.static(path.join(process.cwd(), 'public')));
app.options('/preview', cors());

// Aggressive profanity detection regex (catches leetspeak/obfuscated variants)
const PROFANITY_RE = new RegExp(
  String.raw`\b(?:f[\W_]*u[\W_]*c[\W_]*k+|s[\W_]*h[\W_]*i[\W_]*t+|b[\W_]*i[\W_]*t[\W_]*c[\W_]*h+|n[\W_]*i[\W_]*g+[\W_]*a+)\b`,
  'i'
);

const LEET_MAP = { '@':'a', '$':'s', '0':'o', '1':'i', '!':'i', '3':'e', '4':'a', '5':'s', '7':'t' };
function normToken(s='') {
    s = String(s || '').toLowerCase();
    s = s.replace(/[@$013457!]/g, ch => LEET_MAP[ch] || ch);
    return s.replace(/[^a-z]+/g, '');
}
const hasBad = (txt = '') => {
    return PROFANITY_RE.test(txt) || PROFANITY_RE.test(normToken(txt));
};

const MUTE_PAD = 0.15;  // seconds
const MERGE_GAP = 0.12; // seconds

app.get('/', (_req, res) => res.json({ 
    ok: true, 
    msg: 'FWEA-I Clean Editor API (Optimized)',
    version: '2.0',
    limits: {
        maxFileSize: '50MB',
        maxChunkDuration: '2s',
        supportedFormats: ['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg']
    }
}));

// Enhanced POST /preview with better optimization
app.post('/preview', upload.single('file'), async (req, res) => {
    let tmpFiles = [];
    const startTime = Date.now();
    
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'missing_file' });
        }

        const src = req.file.path;
        const srcSize = getFileSize(src);
        
        console.log(`[upload] ${req.file.originalname} size=${(srcSize/1024/1024).toFixed(1)}MB`);

        // Validate file size client-side
        if (srcSize > 50 * 1024 * 1024) {
            return res.status(413).json({ 
                error: 'file_too_large',
                message: 'File exceeds 50MB limit. Please use a smaller file.',
                fileSize: srcSize,
                maxSize: 50 * 1024 * 1024
            });
        }

        const account = process.env.CF_ACCOUNT_ID || '';
        const token = process.env.CF_API_TOKEN || '';
        
        if (!account || !token) {
            return res.status(500).json({ error: 'cloudflare_credentials_missing' });
        }

        // Get duration with better error handling
        const dur = await ffprobeDuration(src);
        if (dur === 0) {
            return res.status(400).json({ 
                error: 'invalid_audio_file',
                message: 'Could not read audio file. Please check format and try again.'
            });
        }

        console.log(`[audio] duration=${dur.toFixed(1)}s`);

        // More conservative chunking - start smaller
        const INITIAL_CHUNK_SEC = 2; // Start with 2s chunks instead of 5s

        // Enhanced slice creation with multiple optimization levels
        async function makeSlice(start, len, level = 1) {
            const configs = [
                { codec: 'libmp3lame', hz: 32000, br: '64k', ext: 'mp3' },    // Level 1: Standard
                { codec: 'libmp3lame', hz: 24000, br: '48k', ext: 'mp3' },    // Level 2: Reduced
                { codec: 'libmp3lame', hz: 16000, br: '32k', ext: 'mp3' },    // Level 3: Minimal
                { codec: 'libopus', hz: 16000, br: '24k', ext: 'ogg' },       // Level 4: Opus minimal
                { codec: 'libopus', hz: 8000, br: '16k', ext: 'ogg' },        // Level 5: Ultra minimal
            ];

            const config = configs[Math.min(level - 1, configs.length - 1)];
            const out = path.join('uploads', `${Date.now().toString(36)}_${Math.round(start*1000)}_L${level}.${config.ext}`);

            const args = [
                '-hide_banner', '-loglevel', 'error', '-y',
                '-i', src,
                '-ss', String(Math.max(0, start)),
                '-t', String(Math.max(0.1, len)),
                '-ac', '1', // Force mono
                '-ar', String(config.hz),
                '-c:a', config.codec,
            ];

            if (config.codec === 'libopus') {
                args.push('-b:a', config.br, '-vbr', 'on', '-compression_level', '10');
            } else {
                args.push('-b:a', config.br, '-q:a', '9'); // Highest compression for MP3
            }

            args.push(out);

            try {
                await sh(`ffmpeg ${args.map(a => a.includes(' ') ? `"${a}"` : a).join(' ')}`, 15000);
                
                const size = getFileSize(out);
                console.log(`[slice L${level}] ${path.basename(out)} len=${len.toFixed(2)}s size=${(size/1024).toFixed(1)}KB codec=${config.codec}`);
                
                if (size < 4096) {
                    throw new Error(`slice_too_small: ${size} bytes`);
                }
                
                tmpFiles.push(out);
                return { path: out, size, level };
                
            } catch (e) {
                if (fs.existsSync(out)) fs.unlinkSync(out);
                throw e;
            }
        }

        // Enhanced Whisper API call with better error detection
        async function whisperFile(filepath, metadata = {}) {
            const size = getFileSize(filepath);
            
            // Pre-flight size check - be more conservative
            if (size > 5 * 1024 * 1024) { // 5MB pre-flight limit
                throw new Error(`File too large for Whisper API: ${(size/1024/1024).toFixed(1)}MB`);
            }

            const buf = fs.readFileSync(filepath);
            console.log(`[whisper] ${path.basename(filepath)} bytes=${buf.length} level=${metadata.level || 'unknown'}`);

            const fd = new FormData();
            fd.append('input', JSON.stringify({ response_format: 'verbose_json' }));
            
            const mime = filepath.endsWith('.ogg') ? 'audio/ogg' : 'audio/mpeg';
            fd.append('file', new Blob([buf], { type: mime }), path.basename(filepath));

            const url = `https://api.cloudflare.com/client/v4/accounts/${account}/ai/run/@cf/openai/whisper`;
            
            try {
                const cf = await fetch(url, { 
                    method: 'POST', 
                    headers: { Authorization: `Bearer ${token}` }, 
                    body: fd,
                    timeout: 60000 // 60s timeout
                });

                const text = await cf.text();
                let j = {};
                try { j = JSON.parse(text); }
                catch { return { ok: false, error: { message: 'Whisper parse failed', detail: text }, isTooLarge: false, isRateLimit: false, shouldRetry: false }; }

                if (!cf.ok) {
                    const errorInfo = {
                        status: cf.status,
                        body: j,
                        file: path.basename(filepath),
                        bytes: buf.length,
                        level: metadata.level
                    };

                    // Enhanced error detection
                    const errorStr = JSON.stringify(j).toLowerCase();
                    const isTooLarge = /too.large|entity.too.large|payload.too.large|request.too.large/i.test(errorStr);
                    const isRateLimit = cf.status === 429 || /rate.limit|too.many.requests/i.test(errorStr);

                    return { 
                        ok: false, 
                        error: errorInfo,
                        isTooLarge,
                        isRateLimit,
                        shouldRetry: isTooLarge || isRateLimit
                    };
                }

                return { ok: true, json: j };
                
            } catch (fetchError) {
                return { 
                    ok: false, 
                    error: { 
                        message: fetchError.message, 
                        file: path.basename(filepath), 
                        bytes: buf.length,
                        level: metadata.level 
                    },
                    isTooLarge: false,
                    isRateLimit: false,
                    shouldRetry: false
                };
            }
        }

        // Main processing loop with enhanced retry logic
        const segments = [];
        let language = 'auto';
        const total = Math.max(0, dur) || INITIAL_CHUNK_SEC;

        let retryCount = 0;
        const maxRetries = 3;
        let spans = [];

        for (let start = 0; start < total; start += INITIAL_CHUNK_SEC) {
            const len = Math.min(INITIAL_CHUNK_SEC, total - start);
            let currentLen = len;
            let success = false;

            // Try multiple optimization levels for this chunk
            for (let level = 1; level <= 5 && !success; level++) {
                try {
                    // Reduce duration for higher levels
                    if (level > 2) {
                        currentLen = Math.min(len, 1.5); // Max 1.5s for levels 3+
                    }
                    if (level > 4) {
                        currentLen = Math.min(len, 1.0); // Max 1s for level 5
                    }

                    const slice = await makeSlice(start, currentLen, level);
                    const r = await whisperFile(slice.path, { level: slice.level, size: slice.size });

                    if (r.ok) {
                        const result = r.json.result || r.json;
                        if (result.language) language = result.language;

                        const segs = Array.isArray(result.segments) ? result.segments : [];
                        for (const s of segs) {
                            const st = Number(s.start) || 0;
                            const en = Number(s.end) || 0;
                            const txt = s.text || '';
                            segments.push({
                                start: st + start,
                                end: en + start,
                                text: txt
                            });
                            const words = Array.isArray(s.words) ? s.words : [];
                            for (const w of words) {
                                const wtxt = (w.word || '').trim();
                                if (!wtxt) continue;
                                if (hasBad(wtxt)) {
                                    const wst = (Number(w.start) || st) + start;
                                    const wen = (Number(w.end) || en) + start;
                                    spans.push({
                                        start: Math.max(0, wst - MUTE_PAD),
                                        end: Math.max(0, wen + MUTE_PAD),
                                        reason: 'profanity'
                                    });
                                }
                            }
                        }

                        success = true;
                        console.log(`[success] chunk ${start.toFixed(1)}s processed with level ${level}`);

                    } else if (r.shouldRetry && level < 5) {
                        console.log(`[retry] chunk ${start.toFixed(1)}s failed at level ${level}, trying level ${level + 1}`);
                        continue;
                    } else if (r.isRateLimit) {
                        console.log(`[rate-limit] waiting 2s before retry...`);
                        await new Promise(resolve => setTimeout(resolve, 2000));
                        if (retryCount < maxRetries) {
                            retryCount++;
                            level--; // Retry same level
                            continue;
                        }
                    }

                    if (!success && level === 5) {
                        throw new Error(`All optimization levels failed for chunk at ${start.toFixed(1)}s: ${JSON.stringify(r.error)}`);
                    }

                } catch (e) {
                    console.error(`[error] level ${level} failed:`, e.message);
                    if (level === 5) {
                        return res.status(502).json({
                            error: 'whisper_failed',
                            detail: e.message,
                            chunk: { start, len: currentLen, level, retried: true },
                            suggestion: 'Try a shorter audio file or reduce the quality'
                        });
                    }
                }
            }
        }

        // Build profanity spans with word-level precision (if available), else fallback
        if (!spans || !Array.isArray(spans)) { spans = []; }
        if (spans.length === 0) {
            for (const s of segments) {
                if (hasBad(s.text)) {
                    spans.push({
                        start: Math.max(0, s.start - MUTE_PAD),
                        end: Math.max(0, s.end + MUTE_PAD),
                        reason: 'profanity'
                    });
                }
            }
        }

        spans.sort((a, b) => a.start - b.start);

        // Merge overlapping/nearby spans
        const merged = [];
        for (const s of spans) {
            if (!merged.length) {
                merged.push({ ...s });
                continue;
            }
            if (s.start <= merged[merged.length - 1].end + MERGE_GAP) {
                merged[merged.length - 1].end = Math.max(merged[merged.length - 1].end, s.end);
            } else {
                merged.push({ ...s });
            }
        }

        // Enhanced preview generation
        const id = Date.now().toString(36);
        const out = path.join('public', `${id}.mp3`);
        
        const PREVIEW_T = 30;
        const toMute = merged
            .filter(s => s.end > 0 && s.start < PREVIEW_T)
            .map(s => ({ start: Math.max(0, s.start), end: Math.min(PREVIEW_T, s.end) }))
            .filter(s => s.end > s.start);

        // Build audio filter with better precision
        const filter = toMute
            .map(s => `volume=enable='between(t,${s.start.toFixed(3)},${s.end.toFixed(3)})':volume=0`)
            .join(',');

        const args = [
            '-hide_banner', '-loglevel', 'error', '-y',
            '-i', src,
            '-t', String(PREVIEW_T),
            '-ac', '2', // Stereo for preview
            '-ar', '44100', // Standard quality for preview
            '-c:a', 'libmp3lame',
            '-b:a', '192k'
        ];

        if (filter) {
            args.splice(-3, 0, '-af', filter);
        }

        args.push(out);

        try {
            await sh(`ffmpeg ${args.map(a => a.includes(' ') ? `"${a}"` : a).join(' ')}`, 30000);
        } catch (e) {
            console.error('Preview generation failed:', e.message);
            return res.status(500).json({ 
                error: 'preview_generation_failed', 
                detail: e.message 
            });
        }

        const processingTime = Date.now() - startTime;
        const transcript = segments.map(s => s.text || '').join(' ').trim();

        console.log(`[complete] processed in ${(processingTime/1000).toFixed(1)}s, ${segments.length} segments, ${merged.length} muted spans`);

        return res.json({
            preview_url: `/public/${path.basename(out)}`,
            language,
            transcript,
            muted_spans: merged,
            metadata: {
                processingTimeMs: processingTime,
                segmentCount: segments.length,
                originalDuration: dur,
                originalSize: srcSize,
                mutedSpanCount: merged.length
            }
        });

    } catch (err) {
        console.error('Processing error:', err);
        res.status(500).json({ 
            error: 'processing_failed', 
            detail: String(err),
            suggestion: 'Please try with a smaller or different audio file'
        });
    } finally {
        // Enhanced cleanup
        setTimeout(() => {
            try {
                if (req.file?.path && fs.existsSync(req.file.path)) {
                    fs.unlinkSync(req.file.path);
                }
            } catch (e) {
                console.error('Cleanup error (original):', e.message);
            }

            for (const f of tmpFiles) {
                try {
                    if (fs.existsSync(f)) {
                        fs.unlinkSync(f);
                    }
                } catch (e) {
                    console.error('Cleanup error (temp):', e.message);
                }
            }
        }, 5000); // Delay cleanup by 5s to ensure response is sent
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        ok: true,
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        env: {
            hasCloudflareCredentials: !!(process.env.CF_ACCOUNT_ID && process.env.CF_API_TOKEN)
        }
    });
});

// Version marker endpoint for quick sanity checks
app.get('/__version', (_req, res) => {
    res.json({
        ok: true,
        backend: 'node',
        name: 'FWEA-I Clean Editor API (Optimized)',
        version: '2.0',
        build: 'ladder-1 mp3/opus chunks',
        time: new Date().toISOString()
    });
});

app.listen(port, () => {
    console.log(`🚀 Optimized Clean Editor API on http://0.0.0.0:${port}`);
    console.log(`✅ Enhanced file size handling and error management enabled`);
    console.log(`📊 Health check available at /health`);
});

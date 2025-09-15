// backend.js — FWEA-I Clean Editor API (Node/Express)
// Requires: Node 18+ (global fetch/FormData/Blob), ffmpeg, npm i express multer cors

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

const app = express();
const port = process.env.PORT || 8000;

// Upload & static dirs
const upload = multer({ dest: 'uploads/' });
fs.mkdirSync('public', { recursive: true });
fs.mkdirSync('uploads', { recursive: true });

app.use(cors({ origin: '*'}));
app.use('/public', express.static(path.join(process.cwd(), 'public')));

// Basic multilingual profanity set (seed; expand later)
const badWords = [
  /\b(fuck|shit|bitch|asshole|cunt|motherfucker|dick|pussy|nigga|nigger|hoe|slut)\b/i,
  /\b(puta|puto|pendejo|mierda|cabron|coño)\b/i,
  /\b(putain|merde|salope|con)\b/i,
  /\b(bitch|koko|vagin|kaka|manmanw|manman’w)\b/i,
  /\b(merda|caralho|porra|puta)\b/i,
];
const hasBad = (txt='') => badWords.some(rx => rx.test(txt));

app.get('/', (_req,res)=> res.json({ ok:true, msg:'FWEA-I Clean Editor API (Node)' }));

// POST /preview  -> returns { preview_url, language, transcript, muted_spans }
app.post('/preview', upload.single('file'), async (req, res) => {
  let proxy;
  try{
    if(!req.file) return res.status(400).json({ error:'missing_file' });
    const src = req.file.path;

    // Create a small 25–30s Opus proxy for Whisper (keeps body < ~3 MB)
    // We'll try a couple of bitrates if the file is still too large.
    const mkProxy = (brKbps) => new Promise((resolve, reject) => {
      proxy = path.join('uploads', `${Date.now().toString(36)}_proxy.ogg`);
      const args = [
        '-hide_banner','-loglevel','error','-y',
        '-i', src,
        '-t','25',           // a little shorter is fine for detection
        '-ac','1',           // mono
        '-ar','16000',       // 16 kHz
        '-c:a','libopus',    // Opus is very efficient
        '-b:a', `${brKbps}k`,
        '-vbr','on',
        proxy
      ];
      const cmd = `ffmpeg ${args.map(a => a.includes(' ') ? `"${a}"` : a).join(' ')}`;
      exec(cmd, (e, _so, se) => e ? reject(se) : resolve());
    });

    // Try 32k, then 24k, then 16k until size < 3 MB
    const targets = [32, 24, 16];
    let made = false;
    for (const kb of targets) {
      await mkProxy(kb);
      try {
        const sz = fs.statSync(proxy).size;
        if (sz <= 3_000_000) { made = true; break; }
      } catch { /* ignore and fall through */ }
      try { if (proxy && fs.existsSync(proxy)) fs.unlinkSync(proxy); } catch(_){}
    }
    if (!made) {
      // last try at 12k if still too big
      await mkProxy(12);
    }
  
    const account = process.env.CF_ACCOUNT_ID || '';
    const token   = process.env.CF_API_TOKEN  || '';
    if(!account || !token) return res.status(500).json({ error:'cloudflare_credentials_missing' });

    // Cloudflare Whisper call (verbose JSON)
    const input = { response_format: 'verbose_json' };
    const buf = fs.readFileSync(proxy);
    const fd = new FormData();
    fd.append('input', JSON.stringify(input));
    // Use Blob from undici (Node 18+)
    fd.append('file', new Blob([buf], { type: 'audio/ogg' }), path.basename(proxy));

    const url = `https://api.cloudflare.com/client/v4/accounts/${account}/ai/run/@cf/openai/whisper`;
    const cf = await fetch(url, { method:'POST', headers:{ Authorization: `Bearer ${token}` }, body: fd });
    const j = await cf.json();
    if (!cf.ok) {
      console.error('Whisper failed:', cf.status, j);
      return res.status(502).json({ error: 'whisper_failed', detail: j });
    }

    const result = j.result || j;
    const transcript = result.text || '';
    const segs = Array.isArray(result.segments) ? result.segments : [];
    const language = result.language || 'auto';

    // Build spans where a segment contains profanity
    const spans = [];
    for(const s of segs){
      const txt = s.text || '';
      if(hasBad(txt)) spans.push({ start: Number(s.start)||0, end: Number(s.end)||0, reason:'profanity' });
    }
    spans.sort((a,b)=>a.start-b.start);
    const merged=[]; for(const s of spans){ if(!merged.length||s.start>merged[merged.length-1].end) merged.push({...s}); else merged[merged.length-1].end=Math.max(merged[merged.length-1].end,s.end); }

    // FFmpeg: apply mutes and trim to 30s
    const id = Date.now().toString(36);
    const out = path.join('public', `${id}.mp3`);
    const filter = merged.map(s=>`volume=enable='between(t,${Math.max(0,s.start).toFixed(3)},${Math.max(0,s.end).toFixed(3)})':volume=0`).join(',');

    const args = ['-hide_banner','-loglevel','error','-y','-i', proxy, '-t','30'];
    if (filter) args.push('-af', filter);
    args.push('-codec:a','libmp3lame','-b:a','192k', out);

    await new Promise((resolve,reject)=>{
      const cmd = `ffmpeg ${args.map(a=> a.includes(' ')?`"${a}"`:a).join(' ')}`;
      exec(cmd, (e,_so,se)=> e?reject(se):resolve());
    });

    res.json({
      preview_url: `/public/${path.basename(out)}`,
      language,
      transcript,
      muted_spans: merged,
    });
  }catch(err){
    console.error(err);
    res.status(500).json({ error:'preview_failed', detail:String(err) });
  }finally{
    try{ if(req.file?.path) fs.unlink(req.file.path, ()=>{}); }catch(_){ }
    try{ if(proxy && fs.existsSync(proxy)) fs.unlink(proxy, ()=>{}); }catch(_){ }
  }
});

app.listen(port, ()=> console.log(`🚀 Clean Editor API on http://0.0.0.0:${port}`));

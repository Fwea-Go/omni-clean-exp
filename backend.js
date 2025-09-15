// backend.js
import express from "express";
import multer from "multer";
import { exec } from "child_process";
import fetch from "node-fetch";
import fs from "fs";

const app = express();
const port = 8000;

// Multer for file uploads
const upload = multer({ dest: "uploads/" });

// Health check
app.get("/", (req, res) => {
  res.json({ status: "Backend running 🚀" });
});

// Upload route
app.post("/upload", upload.single("audio"), async (req, res) => {
  try {
    const filePath = req.file.path;

    // 1. Send audio to Cloudflare Whisper API
    const whisperResp = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${process.env.CF_ACCOUNT_ID}/ai/run/@cf/openai/whisper`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.CF_API_TOKEN}`,
        },
        body: fs.createReadStream(filePath),
      }
    );

    const whisperData = await whisperResp.json();

    // 2. Scan transcript for profanity (basic example)
    const profanityList = ["fuck", "shit", "bitch", "asshole"];
    const transcript = whisperData.result?.text || "";
    const muteTimes = [];

    profanityList.forEach((badWord) => {
      if (transcript.includes(badWord)) {
        muteTimes.push({ word: badWord, start: 5, end: 8 }); // dummy example
      }
    });

    // 3. Use FFmpeg to mute bad sections (replace with real times)
    const outputPath = `clean-${req.file.filename}.mp3`;
    const filter = muteTimes
      .map(
        (m, i) =>
          `volume=enable='between(t,${m.start},${m.end})':volume=0`
      )
      .join(",");

    const cmd = `ffmpeg -y -i ${filePath} -af "${filter}" ${outputPath}`;

    exec(cmd, (err) => {
      if (err) {
        console.error(err);
        return res.status(500).json({ error: "FFmpeg failed" });
      }

      res.json({
        transcript,
        previewUrl: `/preview/${outputPath}`,
      });
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Upload failed" });
  }
});

// Serve previews
app.use("/preview", express.static("./"));

app.listen(port, () => {
  console.log(`🚀 Backend running on http://0.0.0.0:${port}`);
});

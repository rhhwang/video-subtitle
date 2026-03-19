"""
subtitle_generator.py - Core logic for video transcription and subtitle translation.
"""

import os
import re
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_path: str) -> None:
    """Extract mono 16kHz WAV audio from video using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convert float seconds → SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list) -> str:
    """Convert Whisper segments list to SRT string."""
    parts = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        parts.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(parts) + "\n"


def parse_srt(srt_content: str) -> list[dict]:
    """Parse SRT string into list of {index, timing, text} dicts."""
    blocks = re.split(r"\n\n+", srt_content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            entries.append({
                "index": lines[0],
                "timing": lines[1],
                "text": "\n".join(lines[2:]),
            })
    return entries


def build_srt(entries: list[dict]) -> str:
    """Reconstruct SRT string from list of entries."""
    parts = [f"{e['index']}\n{e['timing']}\n{e['text']}" for e in entries]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Chinese Simplified -> Traditional conversion
# ---------------------------------------------------------------------------

def to_traditional_chinese(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese (Taiwan) using opencc."""
    try:
        import opencc
        converter = opencc.OpenCC("s2twp")  # Simplified -> Traditional (Taiwan + phrases)
        return converter.convert(text)
    except ImportError:
        raise ImportError("opencc-python-reimplemented is required. Install with: pip install opencc-python-reimplemented")


def convert_srt_to_traditional(srt_content: str) -> str:
    """Convert all subtitle text in an SRT from Simplified to Traditional Chinese."""
    entries = parse_srt(srt_content)
    for entry in entries:
        entry["text"] = to_traditional_chinese(entry["text"])
    return build_srt(entries)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_video(
    video_path: str,
    model_size: str = "medium",
    device: str = "cuda",
    language: Optional[str] = None,
    auto_traditional: bool = True,
) -> tuple[str, str]:
    """
    Transcribe a video file using OpenAI Whisper.

    When Chinese is detected and auto_traditional=True (default), the output
    is automatically converted from Simplified to Traditional Chinese.

    Returns:
        (srt_content, detected_language_code)
        Language code will be zh-TW when auto-converted to Traditional Chinese.
    """
    import whisper

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name

    try:
        extract_audio(video_path, audio_path)
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_path, language=language, verbose=False)
        srt = segments_to_srt(result["segments"])
        detected = result.get("language", "unknown")

        # Auto-convert Chinese Simplified -> Traditional
        if auto_traditional and detected == "zh":
            srt = convert_srt_to_traditional(srt)
            detected = "zh-TW"

        return srt, detected
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_srt(
    srt_content: str,
    target_language: str,
    api_key: str,
    batch_size: int = 50,
) -> str:
    """
    Translate SRT subtitle text using OpenAI gpt-4o-mini.
    Preserves all timing information; only translates the text lines.

    Args:
        srt_content:     Original SRT string.
        target_language: Human-readable target language, e.g. "Traditional Chinese".
        api_key:         OpenAI API key.
        batch_size:      Number of subtitle entries per API call.

    Returns:
        Translated SRT string.
    """
    from openai import OpenAI

    entries = parse_srt(srt_content)
    if not entries:
        return srt_content

    client = OpenAI(api_key=api_key)
    all_texts = [e["text"] for e in entries]
    translated_texts: list[str] = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        prompt = (
            f"Translate the following subtitle texts to {target_language}.\n"
            "Rules:\n"
            "- Return ONLY a JSON array with exactly the same number of elements as the input.\n"
            "- Each element is the translated text string.\n"
            "- Preserve line breaks (\\n) inside subtitles if present.\n"
            "- Do not add any explanation or extra keys.\n\n"
            f"Input: {json.dumps(batch, ensure_ascii=False)}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        result = json.loads(raw)
        translated_texts.extend(result)

    # Patch entries with translated text
    for entry, new_text in zip(entries, translated_texts):
        entry["text"] = new_text

    return build_srt(entries)

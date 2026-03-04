"""
cli.py - Command-line interface for video subtitle generation.

Usage examples:
  python cli.py video.mp4
  python cli.py video.mp4 --model large-v3
  python cli.py video.mp4 --translate "Traditional Chinese" --api-key sk-...
  python cli.py video.mp4 --translate "English" --output-dir ./subs
"""

import argparse
import os
import sys
from pathlib import Path

from subtitle_generator import transcribe_video, translate_srt

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from a video file using Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--model",
        default="medium",
        choices=MODEL_CHOICES,
        metavar="SIZE",
        help="Whisper model size (default: medium). Choices: " + ", ".join(MODEL_CHOICES),
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help="Force source language code (e.g. 'zh', 'en'). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--translate",
        default=None,
        metavar="LANGUAGE",
        help="Translate subtitles to this language (e.g. 'Traditional Chinese', 'English').",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        metavar="KEY",
        help="OpenAI API key for translation (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to save SRT files (default: same folder as video).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Whisper inference (default: cuda).",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Transcription ---
    print(f"Loading Whisper '{args.model}' model on {args.device}…")
    print(f"Transcribing: {video_path.name}")

    try:
        srt_content, detected_lang = transcribe_video(
            str(video_path),
            model_size=args.model,
            device=args.device,
            language=args.language,
        )
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return 1

    original_srt = output_dir / f"{video_path.stem}.{detected_lang}.srt"
    original_srt.write_text(srt_content, encoding="utf-8")
    print(f"Saved : {original_srt}")
    print(f"Language detected: {detected_lang}")

    # --- Translation ---
    if args.translate:
        if not args.api_key:
            print(
                "Error: OpenAI API key required for translation.\n"
                "Pass --api-key or set the OPENAI_API_KEY environment variable.",
                file=sys.stderr,
            )
            return 1

        print(f"Translating to: {args.translate}…")
        try:
            translated = translate_srt(srt_content, args.translate, args.api_key)
        except Exception as e:
            print(f"Error during translation: {e}", file=sys.stderr)
            return 1

        lang_slug = (
            args.translate.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        translated_srt = output_dir / f"{video_path.stem}.{lang_slug}.srt"
        translated_srt.write_text(translated, encoding="utf-8")
        print(f"Saved : {translated_srt}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

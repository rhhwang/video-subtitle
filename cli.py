"""
cli.py - Command-line interface for video subtitle generation and translation.

Usage examples:

  # Transcribe a video (auto-detect language)
  python cli.py video.mp4

  # Transcribe with a specific Whisper model
  python cli.py video.mp4 --model large-v3

  # Transcribe + translate in one step
  python cli.py video.mp4 --translate "Traditional Chinese" --api-key sk-...

  # Translate an EXISTING SRT file (skip transcription)
  python cli.py --srt DeBERTa_PPT1.en.srt --translate "Traditional Chinese" --api-key sk-...
  python cli.py --srt DeBERTa_PPT1.en.srt --translate "Japanese" --api-key sk-...

  # Use OPENAI_API_KEY env var instead of --api-key
  set OPENAI_API_KEY=sk-...
  python cli.py --srt subtitle.srt --translate "Korean"
"""

import argparse
import os
import sys
from pathlib import Path

from subtitle_generator import transcribe_video, translate_srt

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or translate SRT subtitle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input: video OR existing SRT (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "video",
        nargs="?",
        help="Path to the input video file (for transcription).",
    )
    input_group.add_argument(
        "--srt",
        metavar="FILE",
        help="Path to an existing SRT file to translate (skips transcription).",
    )

    # --- Transcription options (only relevant when video is given) ---
    parser.add_argument(
        "--model",
        default="medium",
        choices=MODEL_CHOICES,
        metavar="SIZE",
        help="Whisper model size (default: medium). Ignored when --srt is used.",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help="Force source language code (e.g. 'zh', 'en'). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Whisper inference (default: cuda). Ignored when --srt is used.",
    )

    # --- Translation options ---
    parser.add_argument(
        "--translate",
        default=None,
        metavar="LANGUAGE",
        help=(
            "Target language for translation, e.g. 'Traditional Chinese', "
            "'Simplified Chinese', 'Japanese', 'Korean', 'English'."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        metavar="KEY",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable).",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to save output SRT files (default: same folder as input).",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Mode A: translate an existing SRT file
    # -----------------------------------------------------------------------
    if args.srt:
        srt_path = Path(args.srt)
        if not srt_path.exists():
            print(f"Error: SRT file not found: {args.srt}", file=sys.stderr)
            return 1

        if not args.translate:
            print(
                "Error: --translate LANGUAGE is required when using --srt.",
                file=sys.stderr,
            )
            return 1

        if not args.api_key:
            print(
                "Error: OpenAI API key required.\n"
                "Pass --api-key or set the OPENAI_API_KEY environment variable.",
                file=sys.stderr,
            )
            return 1

        output_dir = Path(args.output_dir) if args.output_dir else srt_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        srt_content = srt_path.read_text(encoding="utf-8")
        print(f"Translating '{srt_path.name}' → {args.translate}…")

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
        # Keep original stem, replace/append language tag
        # e.g. DeBERTa_PPT1.en.srt → DeBERTa_PPT1.traditional_chinese.srt
        stem = srt_path.stem  # "DeBERTa_PPT1.en"
        base = stem.rsplit(".", 1)[0] if "." in stem else stem  # "DeBERTa_PPT1"
        output_path = output_dir / f"{base}.{lang_slug}.srt"
        output_path.write_text(translated, encoding="utf-8")
        print(f"Saved : {output_path}")
        print("Done.")
        return 0

    # -----------------------------------------------------------------------
    # Mode B: transcribe a video (+ optionally translate)
    # -----------------------------------------------------------------------
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

    # --- Translation (optional) ---
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

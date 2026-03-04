"""
app.py - Gradio web UI for video subtitle generation.
"""

import os
from pathlib import Path

import gradio as gr

from subtitle_generator import transcribe_video, translate_srt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

TRANSLATION_LANGUAGES = [
    "None",
    "Traditional Chinese (繁體中文)",
    "Simplified Chinese (简体中文)",
    "English",
    "Japanese (日本語)",
    "Korean (한국어)",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Italian",
    "Russian",
    "Arabic",
    "Hindi",
    "Thai",
    "Vietnamese",
    "Indonesian",
]

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Processing function
# ---------------------------------------------------------------------------

def process_video(video_file, model_size, target_language, openai_api_key, progress=gr.Progress()):
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    video_path = Path(video_file)
    # Use a safe ASCII stem to avoid path issues
    stem = video_path.stem

    # --- Transcription ---
    progress(0.05, desc="Loading Whisper model…")
    try:
        srt_content, detected_lang = transcribe_video(
            str(video_path), model_size=model_size
        )
    except Exception as e:
        raise gr.Error(f"Transcription failed: {e}")

    progress(0.65, desc=f"Transcription done (detected: {detected_lang})")

    original_srt_path = OUTPUT_DIR / f"{stem}.{detected_lang}.srt"
    original_srt_path.write_text(srt_content, encoding="utf-8")

    # --- Translation (optional) ---
    translated_srt_path = None

    if target_language and target_language != "None":
        if not openai_api_key:
            raise gr.Error("OpenAI API key is required for translation.")

        progress(0.70, desc=f"Translating to {target_language}…")
        try:
            translated_content = translate_srt(srt_content, target_language, openai_api_key)
        except Exception as e:
            raise gr.Error(f"Translation failed: {e}")

        lang_slug = (
            target_language.split("(")[0].strip().lower().replace(" ", "_")
        )
        translated_srt_path = OUTPUT_DIR / f"{stem}.{lang_slug}.srt"
        translated_srt_path.write_text(translated_content, encoding="utf-8")

    progress(1.0, desc="Done!")

    # Build status message
    lines = [
        f"Detected language : {detected_lang}",
        f"Original SRT      : {original_srt_path.name}",
    ]
    if translated_srt_path:
        lines.append(f"Translated SRT    : {translated_srt_path.name}")

    return (
        "\n".join(lines),
        str(original_srt_path),
        str(translated_srt_path) if translated_srt_path else None,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def toggle_api_key(lang):
    return gr.update(visible=(lang != "None"))


with gr.Blocks(title="Video Subtitle Generator") as demo:
    gr.Markdown("# Video Subtitle Generator")
    gr.Markdown(
        "Upload a video → transcribe with Whisper → download SRT file(s).  \n"
        "Optionally translate to a second language using the OpenAI API."
    )

    with gr.Row():
        # ---- Left column: inputs ----
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")

            model_size = gr.Dropdown(
                choices=MODEL_SIZES,
                value="medium",
                label="Whisper Model Size",
                info="Larger models are more accurate but slower.",
            )

            target_language = gr.Dropdown(
                choices=TRANSLATION_LANGUAGES,
                value="None",
                label="Translation Language (optional)",
            )

            openai_api_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-…",
                type="password",
                visible=False,
            )

            submit_btn = gr.Button("Generate Subtitles", variant="primary", size="lg")

        # ---- Right column: outputs ----
        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Status", lines=5, interactive=False
            )
            original_file = gr.File(label="Original Language Subtitles (.srt)")
            translated_file = gr.File(label="Translated Subtitles (.srt)")

    # Show API key field only when translation is selected
    target_language.change(
        toggle_api_key,
        inputs=target_language,
        outputs=openai_api_key,
    )

    submit_btn.click(
        process_video,
        inputs=[video_input, model_size, target_language, openai_api_key],
        outputs=[status_output, original_file, translated_file],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)

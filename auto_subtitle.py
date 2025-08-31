# auto_subtitle.py

"""
Video Auto-Subtitle Generator using OpenAI's Whisper and FFmpeg.

This script transcribes a video file, generates an SRT subtitle file, and then
uses FFmpeg directly to burn the subtitles into a new video file. This method
is more robust than using MoviePy for subtitle rendering.
"""

import argparse
import os
import whisper
import srt
from datetime import timedelta
import sys
import subprocess # NEW: We will use this to call FFmpeg directly

def generate_subtitles(input_video: str, output_video: str):
    """
    Transcribes a video, generates an SRT file, and burns subtitles into it.
    """
    try:
        # --- 1. Input File Validation ---
        print("Step 1: Validating input file...")
        if not os.path.exists(input_video):
            print(f"Error: Input file not found at '{input_video}'")
            sys.exit(1)
        print(f"Input video found: {input_video}")

        # --- 2. Load Whisper Model ---
        print("\nStep 2: Loading Whisper model (base)...")
        model = whisper.load_model("base")
        print("Whisper model loaded successfully.")

        # --- 3. Transcribe Audio ---
        print(f"\nStep 3: Transcribing audio from '{os.path.basename(input_video)}'...")
        print("This may take some time depending on the video length...")
        result = model.transcribe(input_video, fp16=False)
        print("Transcription complete.")

        # --- 4. Generate SRT Subtitle File ---
        print("\nStep 4: Generating SRT subtitle file...")
        subtitles = []
        for i, segment in enumerate(result["segments"], start=1):
            start_time = timedelta(seconds=segment['start'])
            end_time = timedelta(seconds=segment['end'])
            text = segment['text'].strip()
            subtitle = srt.Subtitle(index=i, start=start_time, end=end_time, content=text)
            subtitles.append(subtitle)

        srt_content = srt.compose(subtitles)
        output_dir = os.path.dirname(output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        srt_filename = os.path.splitext(os.path.basename(output_video))[0] + ".srt"
        srt_path = os.path.join(output_dir if output_dir else '.', srt_filename)
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"SRT file saved to: {srt_path}")

        # --- 5. Burn Subtitles into Video using FFmpeg ---
        print(f"\nStep 5: Burning subtitles into '{os.path.basename(output_video)}' using FFmpeg...")

        # Get absolute paths to be safe
        input_video_path = os.path.abspath(input_video)
        srt_path_abs = os.path.abspath(srt_path)
        output_video_path = os.path.abspath(output_video)

        # For Windows, FFmpeg's subtitles filter needs a specially escaped path
        escaped_srt_path = srt_path_abs.replace('\\', '/').replace(':', '\\:')

        # Construct the FFmpeg command
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', input_video_path,
            '-vf', f"subtitles='{escaped_srt_path}'",
            '-c:a', 'copy',  # Copy audio stream without re-encoding
            output_video_path
        ]
        
        # Execute the command
        process = subprocess.run(command, capture_output=True, text=True)

        if process.returncode != 0:
            print("\n❌ FFmpeg Error:")
            print(process.stderr)
            sys.exit(1)

        print("\n-------------------------------------------")
        print("✅ Done! Process completed successfully.")
        print(f"Output video saved to: {output_video}")
        print("-------------------------------------------")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred.")
        print(f"Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and burn subtitles into a video using OpenAI's Whisper."
    )
    parser.add_argument(
        "input_video", type=str, help="Path to the input video file."
    )
    parser.add_argument(
        "output_video", type=str, help="Path for the output video file."
    )
    args = parser.parse_args()
    generate_subtitles(args.input_video, args.output_video)

    

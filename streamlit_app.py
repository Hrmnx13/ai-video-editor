import streamlit as st
import os
import subprocess
import whisper
import srt
from datetime import timedelta
import uuid
import tempfile

# --- Helper Functions (from our previous backend) ---

def generate_srt(result, output_path):
    """Generates an SRT subtitle file from Whisper's transcription result."""
    subtitles = []
    for i, segment in enumerate(result["segments"], start=1):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        text = segment['text'].strip()
        subtitle = srt.Subtitle(index=i, start=start_time, end=end_time, content=text)
        subtitles.append(subtitle)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))
    st.sidebar.success(f"SRT file generated.")

def run_ffmpeg_command(command, step_name):
    """Runs an FFmpeg command and raises an error if it fails."""
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        st.sidebar.info(f"FFmpeg logs for {step_name}:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg failed during '{step_name}'.")
        st.code(e.stderr)
        raise RuntimeError(f"FFmpeg failed during '{step_name}'. Error: {e.stderr}")

# --- Streamlit UI Configuration ---

st.set_page_config(
    page_title="AI Video Editor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 AI Video Editor")
st.markdown("Automatically add subtitles, trim, and watermark your videos with ease.")

# --- Main Application Logic ---

# Use a temporary directory to store all files for a session
temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = temp_dir.name

# 1. Video Upload
uploaded_video = st.file_uploader(
    "Upload Your Video",
    type=["mp4", "mov", "avi", "webm"],
    help="Drag and drop or click to upload your video file."
)

if uploaded_video:
    # Save the uploaded video to a temporary file
    video_path = os.path.join(temp_dir_path, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    st.video(video_path)
    
    # --- Sidebar for Editing Options ---
    st.sidebar.header("Editing Options")
    
    # Watermark Options
    add_watermark = st.sidebar.checkbox("Add Watermark")
    watermark_path = None
    if add_watermark:
        uploaded_watermark = st.sidebar.file_uploader("Upload Watermark (PNG)", type=["png"])
        if uploaded_watermark:
            watermark_path = os.path.join(temp_dir_path, uploaded_watermark.name)
            with open(watermark_path, "wb") as f:
                f.write(uploaded_watermark.getbuffer())
            st.sidebar.image(watermark_path, width=100)

    # Trim Options
    trim_video = st.sidebar.checkbox("Trim Video")
    start_time, end_time = None, None
    if trim_video:
        start_time = st.sidebar.time_input("Start Time", value=timedelta(seconds=0))
        end_time = st.sidebar.time_input("End Time", value=timedelta(seconds=30))

    # Resolution Options
    resolution = st.sidebar.selectbox(
        "Change Resolution",
        options=["Keep Original", "1080p (1920x1080)", "720p (1280x720)", "480p (854x480)"],
        index=0
    )
    resolution_map = {
        "1080p (1920x1080)": "1080p",
        "720p (1280x720)": "720p",
        "480p (854x480)": "480p",
        "Keep Original": "original"
    }
    resolution_val = resolution_map[resolution]

    # --- Process Button ---
    if st.button("✨ Create My Video", use_container_width=True):
        with st.spinner("Processing your video... This might take a few minutes."):
            try:
                # STEP 1: Transcribe and generate SRT
                st.sidebar.info("Transcribing audio...")
                model = whisper.load_model("base")
                result = model.transcribe(video_path, fp16=False)
                
                srt_filename = f"{uuid.uuid4()}.srt"
                srt_path = os.path.join(temp_dir_path, srt_filename)
                generate_srt(result, srt_path)

                # STEP 2: Start building the FFmpeg command
                current_video_input = video_path
                
                # A: Trim if requested (creates a temporary trimmed file)
                if trim_video:
                    trimmed_filename = f"trimmed_{uuid.uuid4()}.mp4"
                    trimmed_path = os.path.join(temp_dir_path, trimmed_filename)
                    
                    trim_cmd = ['ffmpeg', '-y', '-i', current_video_input,
                                '-ss', str(start_time), '-to', str(end_time),
                                '-c', 'copy', trimmed_path]
                    run_ffmpeg_command(trim_cmd, "Trimming")
                    current_video_input = trimmed_path

                # B: Apply visual filters (subtitles, watermark, resolution)
                processed_filename = f"processed_{uploaded_video.name}"
                output_path = os.path.join(temp_dir_path, processed_filename)
                escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
                
                visual_cmd = ['ffmpeg', '-y', '-i', current_video_input]
                if watermark_path:
                    visual_cmd.extend(['-i', watermark_path])

                video_filters = []
                if resolution_val == '1080p': video_filters.append("scale=1920:-2")
                elif resolution_val == '720p': video_filters.append("scale=1280:-2")
                elif resolution_val == '480p': video_filters.append("scale=854:-2")
                
                video_filters.append(f"subtitles='{escaped_srt_path}'")
                
                if watermark_path:
                    watermark_scale = "scale=200:-1"
                    if resolution_val == '1080p': watermark_scale = "scale=250:-1"
                    elif resolution_val == '720p': watermark_scale = "scale=150:-1"
                    elif resolution_val == '480p': watermark_scale = "scale=100:-1"
                    
                    filter_chain = f"[1:v]{watermark_scale}[wm];[0:v]{','.join(video_filters)}[vid];[vid][wm]overlay=10:10"
                    visual_cmd.extend(['-filter_complex', filter_chain])
                else:
                    visual_cmd.extend(['-vf', ",".join(video_filters)])

                visual_cmd.extend(['-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p'])
                visual_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
                visual_cmd.append(output_path)
                
                run_ffmpeg_command(visual_cmd, "Applying Visual Filters")

                st.success("🎉 Video processed successfully!")
                
                # Provide the download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Video",
                        data=file,
                        file_name=processed_filename,
                        mime="video/mp4",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Clean up the temporary directory
                temp_dir.cleanup()

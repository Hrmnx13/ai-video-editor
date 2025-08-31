import os
import subprocess
import whisper
import srt
from datetime import timedelta
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid # To create unique temporary filenames

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'png'}

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='.', static_url_path='')

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# --- Helper Functions ---
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_srt(result, output_path):
    subtitles = []
    for i, segment in enumerate(result["segments"], start=1):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        text = segment['text'].strip()
        subtitle = srt.Subtitle(index=i, start=start_time, end=end_time, content=text)
        subtitles.append(subtitle)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))
    print(f"SRT file generated at: {output_path}")

def run_ffmpeg_command(command, step_name):
    """A helper function to run FFmpeg commands and handle errors."""
    print(f"Running FFmpeg for: {step_name}")
    print("Command:", " ".join(command))
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"FFmpeg {step_name} successful.")
        print("FFmpeg logs:", process.stderr)
    except subprocess.CalledProcessError as e:
        # Raise a new exception with a clear error message
        raise RuntimeError(f"FFmpeg failed during '{step_name}'. Error: {e.stderr}")

# --- API Routes ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process_video():
    # --- 1. Handle File Uploads and Options ---
    if 'videoFile' not in request.files: return jsonify({'error': 'No video file part'}), 400
    video_file = request.files['videoFile']
    if video_file.filename == '' or not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS): return jsonify({'error': 'Invalid video file'}), 400

    unique_id = str(uuid.uuid4())
    video_filename = f"{unique_id}_{secure_filename(video_file.filename)}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video_file.save(video_path)
    print(f"Video saved to: {video_path}")
    
    # Store all temporary files to be cleaned up later
    temp_files = [video_path]

    try:
        watermark_path = None
        if 'watermark' in request.form and request.form['watermark'] == 'true':
            if 'watermarkFile' not in request.files: return jsonify({'error': 'Watermark was checked but no file was provided'}), 400
            watermark_file = request.files['watermarkFile']
            if watermark_file and allowed_file(watermark_file.filename, ALLOWED_IMAGE_EXTENSIONS):
                watermark_filename = f"{unique_id}_{secure_filename(watermark_file.filename)}"
                watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename)
                watermark_file.save(watermark_path)
                temp_files.append(watermark_path)
                print(f"Watermark saved to: {watermark_path}")

        # --- 2. Transcribe and Generate Subtitles ---
        print("Loading Whisper model and transcribing...")
        model = whisper.load_model("base")
        result = model.transcribe(video_path, fp16=False)
        
        srt_filename = f"{unique_id}.srt"
        srt_path = os.path.join(app.config['UPLOAD_FOLDER'], srt_filename)
        generate_srt(result, srt_path)
        temp_files.append(srt_path)

        # --- 3. Start Sequential FFmpeg Processing ---
        current_video_input = video_path
        
        # STEP A: Trim the video (if requested)
        if 'trim' in request.form and request.form['trim'] == 'true':
            trimmed_filename = f"trimmed_{video_filename}"
            trimmed_path = os.path.join(app.config['UPLOAD_FOLDER'], trimmed_filename)
            temp_files.append(trimmed_path)
            
            trim_cmd = ['ffmpeg', '-y', 
                        '-i', current_video_input,
                        '-ss', request.form['startTime'], 
                        '-to', request.form['endTime'],
                        '-c', 'copy', # Very fast, just copies the streams
                        trimmed_path]
            run_ffmpeg_command(trim_cmd, "Trimming")
            current_video_input = trimmed_path

        # STEP B: Apply Visual Filters (Subtitles, Watermark, Resolution)
        processed_filename = f"processed_{video_filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
        
        visual_cmd = ['ffmpeg', '-y', '-i', current_video_input]
        if watermark_path:
            visual_cmd.extend(['-i', watermark_path])

        resolution = request.form.get('resolution', 'original')
        video_filters = []
        if resolution == '1080p': video_filters.append("scale=1920:-2")
        elif resolution == '720p': video_filters.append("scale=1280:-2")
        elif resolution == '480p': video_filters.append("scale=854:-2")
        
        video_filters.append(f"subtitles='{escaped_srt_path}'")
        
        if watermark_path:
            watermark_scale = "scale=200:-1"
            if resolution == '1080p': watermark_scale = "scale=250:-1"
            elif resolution == '720p': watermark_scale = "scale=150:-1"
            elif resolution == '480p': watermark_scale = "scale=100:-1"
            
            filter_chain = f"[1:v]{watermark_scale}[wm];[0:v]{','.join(video_filters)}[vid];[vid][wm]overlay=10:10"
            visual_cmd.extend(['-filter_complex', filter_chain])
        else:
            visual_cmd.extend(['-vf', ",".join(video_filters)])

        visual_cmd.extend(['-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p'])
        visual_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        visual_cmd.append(output_path)
        
        run_ffmpeg_command(visual_cmd, "Applying Visual Filters")

        return jsonify({'processed_filename': processed_filename}), 200

    except Exception as e:
        # If any step fails, return the error
        print(f"An error occurred during processing: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # STEP C: Clean up all temporary files
        print("Cleaning up temporary files...")
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed: {f}")

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)


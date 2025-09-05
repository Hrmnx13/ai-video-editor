import os
import subprocess
import whisper
import srt
from datetime import timedelta
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from textblob import TextBlob
from rake_nltk import Rake

# Imports for the Sumy summarizer library
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'png'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac'}

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

def generate_srt_and_get_text(result, output_path):
    full_text = ""
    subtitles = []
    for i, segment in enumerate(result["segments"], start=1):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        text = segment['text'].strip()
        full_text += text + " "
        subtitle = srt.Subtitle(index=i, start=start_time, end=end_time, content=text)
        subtitles.append(subtitle)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))
    print(f"SRT file generated at: {output_path}")
    return full_text.strip()

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1: return "Positive"
    elif polarity < -0.1: return "Negative"
    else: return "Neutral"

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:5]

def create_summary(text, language="english", sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    summary_sentences = summarizer(parser.document, sentences_count)
    summary = " ".join([str(sentence) for sentence in summary_sentences])
    if not summary:
        return "Could not generate a summary. The video might be too short."
    return summary

def generate_ai_titles(summary, keywords):
    titles = []
    if not keywords:
        return ["A General Overview", "Interesting Discussion Points", "Video Analysis"]
    titles.append(f"A Deep Dive into {keywords[0].title()}")
    if len(keywords) > 1:
        titles.append(f"Exploring {keywords[0].title()} and {keywords[1]}")
    titles.append(f"What You Need to Know About {keywords[0].title()}")
    return titles

def find_silences(transcription_result, min_silence_duration=2.0):
    silences = []
    segments = transcription_result['segments']
    if not segments: return []
    if segments[0]['start'] > min_silence_duration:
         silences.append(f"0:00 to {timedelta(seconds=int(segments[0]['start']))}")
    for i in range(len(segments) - 1):
        end_of_current = segments[i]['end']
        start_of_next = segments[i+1]['start']
        silence_duration = start_of_next - end_of_current
        if silence_duration > min_silence_duration:
            start_time = timedelta(seconds=int(end_of_current))
            end_time = timedelta(seconds=int(start_of_next))
            silences.append(f"{start_time} to {end_time}")
    if not silences:
        return ["No significant silences found."]
    return silences

# NEW: AI function to interpret color grading prompts
def get_color_values_from_prompt(prompt):
    """Simulates an AI interpreting a color prompt."""
    prompt = prompt.lower()
    # (brightness, contrast, saturation)
    styles = {
        "vintage": (-0.1, 0.9, 0.8),
        "cinematic": (-0.05, 1.1, 1.2),
        "dark": (-0.2, 1.2, 1.0),
        "moody": (-0.15, 1.15, 0.9),
        "warm": (0.0, 1.0, 1.3),
        "sunny": (0.1, 1.1, 1.2),
        "bright": (0.15, 1.0, 1.1),
        "vibrant": (0.0, 1.1, 1.5)
    }
    for key, values in styles.items():
        if key in prompt:
            print(f"AI detected style '{key}', applying color values: {values}")
            return values
    print("No specific AI style detected, using default color values.")
    return (0.0, 1.0, 1.0) # Default values

def run_ffmpeg_command(command, step_name):
    print(f"Running FFmpeg for: {step_name}")
    print("Command:", " ".join(command))
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"FFmpeg {step_name} successful.")
        print("FFmpeg logs:", process.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed during '{step_name}'. Error: {e.stderr}")

# --- API Routes ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process_video():
    if 'videoFile' not in request.files: return jsonify({'error': 'No video file found'}), 400
    video_file = request.files['videoFile']
    if video_file.filename == '' or not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS): return jsonify({'error': 'Invalid video file'}), 400

    unique_id = str(uuid.uuid4())
    video_filename = f"{unique_id}_{secure_filename(video_file.filename)}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video_file.save(video_path)
    
    temp_files = [video_path]

    try:
        # Handle file uploads
        watermark_path, music_path = None, None
        if 'watermark' in request.form and request.form['watermark'] == 'true':
            if 'watermarkFile' in request.files and allowed_file(request.files['watermarkFile'].filename, ALLOWED_IMAGE_EXTENSIONS):
                watermark_file = request.files['watermarkFile']
                watermark_filename = f"{unique_id}_{secure_filename(watermark_file.filename)}"
                watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename)
                watermark_file.save(watermark_path)
                temp_files.append(watermark_path)
        
        if 'addMusic' in request.form and request.form['addMusic'] == 'true':
            if 'musicFile' in request.files and allowed_file(request.files['musicFile'].filename, ALLOWED_AUDIO_EXTENSIONS):
                music_file = request.files['musicFile']
                music_filename = f"{unique_id}_{secure_filename(music_file.filename)}"
                music_path = os.path.join(app.config['UPLOAD_FOLDER'], music_filename)
                music_file.save(music_path)
                temp_files.append(music_path)

        # AI Processing
        model = whisper.load_model("base")
        result = model.transcribe(video_path, fp16=False)
        srt_filename = f"{unique_id}.srt"
        srt_path = os.path.join(app.config['UPLOAD_FOLDER'], srt_filename)
        full_transcript = generate_srt_and_get_text(result, srt_path)
        temp_files.append(srt_path)

        sentiment = analyze_sentiment(full_transcript)
        keywords = extract_keywords(full_transcript)
        summary = create_summary(full_transcript)
        titles = generate_ai_titles(summary, keywords)
        silences = find_silences(result)

        # Video Processing
        current_video_input = video_path
        if 'trim' in request.form and request.form['trim'] == 'true':
            trimmed_filename = f"trimmed_{video_filename}"
            trimmed_path = os.path.join(app.config['UPLOAD_FOLDER'], trimmed_filename)
            temp_files.append(trimmed_path)
            trim_cmd = ['ffmpeg', '-y', '-i', current_video_input, '-ss', request.form['startTime'], '-to', request.form['endTime'], '-c:v', 'libx264', '-c:a', 'aac', trimmed_path]
            run_ffmpeg_command(trim_cmd, "Trimming")
            current_video_input = trimmed_path

        processed_filename = f"processed_{secure_filename(video_file.filename)}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
        
        visual_cmd = ['ffmpeg', '-y', '-i', current_video_input]
        if watermark_path: visual_cmd.extend(['-i', watermark_path])
        if music_path: visual_cmd.extend(['-i', music_path])

        video_filters = []
        rotation = request.form.get('rotation', '0')
        if rotation == '90': video_filters.append("transpose=1")
        elif rotation == '-90': video_filters.append("transpose=2")
        
        # NEW: Use AI prompt for color values
        color_prompt = request.form.get('colorPrompt', '')
        brightness, contrast, saturation = get_color_values_from_prompt(color_prompt)
        if brightness != 0.0 or contrast != 1.0 or saturation != 1.0: 
            video_filters.append(f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}")

        video_filters.append(f"subtitles='{escaped_srt_path}'")
        
        video_chain = f"[0:v]{','.join(video_filters)}[v_out];"
        
        if watermark_path and music_path:
            watermark_chain = f"[1:v]scale=120:-1[wm];"
            overlay_chain = f"[v_out][wm]overlay=10:10[v];"
            music_volume = float(request.form.get('musicVolume', 0.5))
            audio_mix_chain = f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[main_a];[2:a]volume={music_volume}[music_a];[main_a][music_a]amix=inputs=2:duration=longest[a]"
            filter_complex = f"{video_chain}{watermark_chain}{overlay_chain}{audio_mix_chain}"
            visual_cmd.extend(['-filter_complex', filter_complex, '-map', '[v]', '-map', '[a]'])
        elif watermark_path:
            watermark_chain = f"[1:v]scale=120:-1[wm];"
            overlay_chain = f"[v_out][wm]overlay=10:10[v]"
            filter_complex = f"{video_chain}{watermark_chain}{overlay_chain}"
            visual_cmd.extend(['-filter_complex', filter_complex, '-map', '[v]', '-map', '0:a'])
        elif music_path:
            music_volume = float(request.form.get('musicVolume', 0.5))
            audio_mix_chain = f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[main_a];[1:a]volume={music_volume}[music_a];[main_a][music_a]amix=inputs=2:duration=longest[a]"
            filter_complex = f"{video_chain}{audio_mix_chain}"
            visual_cmd.extend(['-filter_complex', filter_complex, '-map', '[v_out]', '-map', '[a]'])
        else:
            visual_cmd.extend(['-vf', ",".join(video_filters)])

        quality = request.form.get('quality', 'medium')
        crf_value = '23'
        if quality == 'high': crf_value = '18'
        elif quality == 'low': crf_value = '28'

        visual_cmd.extend(['-c:v', 'libx264', '-crf', crf_value, '-preset', 'medium', '-pix_fmt', 'yuv420p'])
        if not music_path: visual_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        visual_cmd.append(output_path)
        
        run_ffmpeg_command(visual_cmd, "Applying Visual Filters")

        return jsonify({
            'processed_filename': processed_filename,
            'sentiment': sentiment,
            'keywords': keywords,
            'summary': summary,
            'titles': titles,
            'silences': silences
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    import nltk
    print("Downloading necessary NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('brown')
    try:
        nltk.download('punkt_tab')
    except:
        print("'punkt_tab' not found, skipping.")
    nltk.download('averaged_perceptron_tagger') 
    print("NLTK data download complete.")
    # CORRECTED: Run the server on your network
    app.run(host='0.0.0.0', port=5000, debug=True)
    


import warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
import os, base64, io, random, datetime, psycopg2, tempfile, torch, re
import whisper
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai
from gtts import gTTS 
from que import generate_answer 
from src.helper import normalize_amharic 
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "medical_secure_key_123")

# --- HYBRID STT INITIALIZATION ---
print("⏳ Loading Hybrid STT Models (Small-Am Local & Medium English)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Local Amharic Model (Whisper-small-am)
AMHARIC_MODEL_PATH = r'C:\Users\Belayneh\Downloads\MedicalChatbot\model'
am_stt_pipe = pipeline(
    "automatic-speech-recognition",
    model=AMHARIC_MODEL_PATH,
    chunk_length_s=30,
    device=device,
    use_fast=False
)

# 2. English Model (Whisper-medium)
en_stt_model = whisper.load_model("medium")
print("✅ Hybrid STT Models Loaded Successfully!")

# --- MAIL CONFIGURATION ---
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_SSL=True,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "belzman2011@gmail.com"),
    MAIL_PASSWORD=os.getenv("uliegqopvowpzyll"), 
    MAIL_DEFAULT_SENDER=os.getenv("MAIL_USERNAME", "belzman2011@gmail.com")
)
mail = Mail(app)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_db_connection():
    return psycopg2.connect(
        dbname="Chat",
        user="postgres",
        password="Bel123",
        host="localhost"
    )

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/auth/register', methods=['POST'])
def register():
    data = request.json
    username, email, password = data.get('username'), data.get('email'), data.get('password')
    if not username or not email or not password:
        return jsonify({"success": False, "message": "All fields are required"}), 400
    
    hashed_pw = generate_password_hash(password)
    verification_code = str(random.randint(100000, 999999))
    expiry = datetime.datetime.now() + datetime.timedelta(minutes=15)
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM login WHERE username = %s OR email = %s", (username, email))
        if cur.fetchone(): 
            return jsonify({"success": False, "message": "User already exists"}), 400
        
        cur.execute("INSERT INTO login (username, email, password_hash, reset_code, code_expiry) VALUES (%s, %s, %s, %s, %s)",
            (username, email, hashed_pw, verification_code, expiry))
        conn.commit()
        
        msg = Message("Verify Your SmartHIVCare Account", recipients=[email])
        msg.body = f"Hello {username},\n\nYour registration verification code is: {verification_code}"
        mail.send(msg)
        return jsonify({"success": True, "message": "Code sent to email"})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        cur.close(); conn.close()

@app.route('/auth/login', methods=['POST'])
def handle_login():
    data = request.json
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT id, password_hash, username FROM login WHERE username = %s AND reset_code IS NULL", (data['username'],))
    user = cur.fetchone(); cur.close(); conn.close()
    if user and check_password_hash(user[1], data['password']):
        session['user_id'], session['username'] = user[0], user[2]
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Invalid credentials or account not verified."})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: 
        return redirect(url_for('index'))
    return render_template('index.html', username=session.get('username'))

@app.route('/logout')
def logout():
    # Clear the session and redirect to login page
    session.clear()
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask():
    try:
        lang = request.form.get('language', 'English')
        answer_mode = request.form.get('answer_mode') 
        
        # --- STT (Speech to Text) Processing ---
        if 'question' in request.files and request.files['question'].filename != '':
            audio_file = request.files['question']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                audio_file.save(tmp_audio.name)
                tmp_path = tmp_audio.name
            
            if lang.lower() == "amharic":
                # Speech-Text (Amharic) via Whisper-small-am
                result = am_stt_pipe(tmp_path, generate_kwargs={"language": "am"})
                user_input = normalize_amharic(result["text"].strip())
            else:
                # Speech-Text (English) via Whisper-medium
                result = en_stt_model.transcribe(tmp_path, language="en", fp16=False)
                user_input = result["text"].strip()
            
            os.remove(tmp_path)
        else:
            # Text-Text input
            user_input = request.form.get('question')

        if not user_input: 
            return jsonify({"error": "No input received"}), 400

        # --- RAG Answer Generation ---
        answer = generate_answer(user_input, language=lang)
        audio_base64 = "" 

        # --- TTS (Text to Speech) Processing ---
        if answer_mode == "Voice" and answer:
            try:
                tts_lang = 'am' if lang.lower() == 'amharic' else 'en'
                clean_text = re.sub(r'[^\w\s.,!?]', '', answer.split("📚")[0].strip())
                
                tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                audio_base64 = base64.b64encode(fp.read()).decode('utf-8')
            except Exception as tts_err:
                print(f"TTS Error: {tts_err}")

        return jsonify({
            "answer": answer, 
            "transcription": user_input, 
            "audio": audio_base64
        })

    except Exception as e:
        print(f"❌ Error in /ask route: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SmartHIVCare Started with Hybrid Local STT")
    print("🔗 Access at http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)


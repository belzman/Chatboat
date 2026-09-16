import os
import base64
import io
import random
import datetime
import psycopg2
import whisper
import tempfile
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai
from google.genai import types
from gtts import gTTS 
from que import generate_answer 
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "medical_secure_key_123")

print("⏳ Loading Whisper 'medium' Model (Approx 461MB)...")
# Using medium model for better Amharic phonetic understanding
stt_model = whisper.load_model("medium") 
print("✅ Whisper medium Loaded Successfully!")

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

@app.route('/auth/verify-registration', methods=['POST'])
def verify_registration():
    data = request.json
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT id FROM login WHERE username = %s AND reset_code = %s AND code_expiry > %s",
        (data.get('username'), data.get('code'), datetime.datetime.now()))
    user = cur.fetchone()
    if user:
        cur.execute("UPDATE login SET reset_code = NULL, code_expiry = NULL WHERE id = %s", (user[0],))
        conn.commit(); success = True
    else: success = False
    cur.close(); conn.close()
    return jsonify({"success": success})

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

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('index'))
    return render_template('index.html', username=session.get('username'))

@app.route('/ask', methods=['POST'])
def ask():
    try:
        lang = request.form.get('language', 'English')
        answer_mode = request.form.get('answer_mode') 
        
        if 'question' in request.files and request.files['question'].filename != '':
            audio_file = request.files['question']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                audio_file.save(tmp_audio.name)
                tmp_path = tmp_audio.name
            
            whisper_lang = "am" if lang.lower() == "amharic" else "en"
       
            # UPDATED: Strict parameters to stop Amharic hallucinations
            result = stt_model.transcribe(
                tmp_path, 
                language=whisper_lang, 
                fp16=False, 
                beam_size=5,
                best_of=5,
                temperature=0.0,  # Forces most likely token, reducing "junk" text
                condition_on_previous_text=False # Prevents looping errors
            )
            user_input = result["text"].strip()
            
            os.remove(tmp_path)
        else:
            user_input = request.form.get('question')

        if not user_input: 
            return jsonify({"error": "No input received"}), 400

        # Generate RAG answer
        answer = generate_answer(user_input, language=lang)
        audio_base64 = "" 

        # Voice output handling
        if answer_mode == "Voice" and answer:
            try:
                tts_lang = 'am' if lang.lower() == 'amharic' else 'en'
                # Remove markdown formatting for cleaner speech
                clean_text = answer.split("📚")[0].strip().replace("*", "").replace("#", "")
                
                # Using gTTS for high-quality Amharic synthesis as requested
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SmartHIVCare Started with Enhanced Amharic STT")
    print("🔗 Access at http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
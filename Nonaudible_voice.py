import os
import base64
import io
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from gtts import gTTS  # Alternative stable TTS
from que import generate_answer
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        lang = request.form.get('language', 'English')
        answer_mode = request.form.get('answer_mode')
        
        # 1. Bimodal Input Handling (Voice-to-Text)
        if 'question' in request.files and request.files['question'].filename != '':
            audio_data = request.files['question'].read()
            trans_res = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    f"Transcribe this medical question accurately in {lang}:", 
                    types.Part.from_bytes(data=audio_data, mime_type='audio/wav')
                ]
            )
            user_input = trans_res.text.strip()
        else:
            user_input = request.form.get('question')

        if not user_input:
            return jsonify({"error": "No input received"}), 400

        # 2. RAG-LLM Core Logic
        answer = generate_answer(user_input, language=lang)

        # 3. Bimodal Output Handling (Text-to-Voice)
        audio_base64 = "" 
        if answer_mode == "Voice":
            # --- SOLUTION A: Native Gemini Multimodal Audio ---
            try:
                print(f" Native Audio Synthesis ({lang})...")
                tts_res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[f"Read this naturally in {lang}: {answer}"],
                    config=types.GenerateContentConfig(response_modalities=["AUDIO"])
                )
                
                if tts_res.candidates and tts_res.candidates[0].content.parts:
                    for part in tts_res.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            audio_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                            print("✅ Native Audio Success")
                            break
            
            except Exception as native_err:
                print(f"⚠️ Native Audio failed, switching to Alternative (gTTS)...")
                
            # --- SOLUTION B: Alternative gTTS Fallback (Guaranteed to work) ---
            if not audio_base64:
                try:
                    # Map languages for gTTS
                    tts_lang = 'am' if lang.lower() == 'amharic' else 'en'
                    
                    # Remove citations/source tags from audio so it sounds natural
                    clean_answer = answer.split("📚")[0].strip() 
                    
                    tts = gTTS(text=clean_answer, lang=tts_lang)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    audio_base64 = base64.b64encode(fp.read()).decode('utf-8')
                    print("✅ Alternative Audio (gTTS) Success")
                except Exception as gtts_err:
                    print(f"❌ Both Audio methods failed: {gtts_err}")

        return jsonify({
            "answer": answer, 
            "transcription": user_input, 
            "audio": audio_base64
        })

    except Exception as e:
        print(f"🛑 Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
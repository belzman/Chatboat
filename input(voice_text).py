import os
import base64
import time
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from que import generate_answer
from dotenv import load_dotenv

# 1. Initialization
load_dotenv()
app = Flask(__name__)

# Use the Gemini 2.5 Flash Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@app.route('/')
def index():
    """Serves the main dashboard."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        lang = request.form.get('language', 'English')
        answer_mode = request.form.get('answer_mode') # 'Text' or 'Voice'
        
        # --- INPUT MODALITY HANDLING (V-x or T-x) ---
        if 'question' in request.files and request.files['question'].filename != '':
            audio_data = request.files['question'].read()
            print(f"🎤 Processing Voice Input ({lang})...")
            
            # Transcription via Multimodal Part
            trans_res = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    f"Transcribe this medical question accurately in {lang}. Output ONLY the text:", 
                    types.Part.from_bytes(data=audio_data, mime_type='audio/wav')
                ]
            )
            user_input = trans_res.text.strip()
        else:
            user_input = request.form.get('question')

        if not user_input:
            return jsonify({"error": "No input received"}), 400

        # --- CORE RAG ENGINE ---
        # Retrieval from your medical PDFs via que.py
        print(f"🧠 Querying Knowledge Base: {user_input[:50]}...")
        answer = generate_answer(user_input, language=lang)

        # --- OUTPUT MODALITY HANDLING (x-V or x-T) ---
        audio_base64 = "" 
        
        if answer_mode == "Voice":
            print(f"🔊 Generating Native Voice Output ({lang})...")
            try:
                # 2026 STABLE FIX: Use response_mime_type instead of response_modalities
                tts_res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[f"Read this medical advice naturally in {lang}: {answer}"],
                    config=types.GenerateContentConfig(
                        response_mime_type="audio/wav" 
                    )
                )
                
                # Extract audio data from parts
                if tts_res.candidates and tts_res.candidates[0].content.parts:
                    for part in tts_res.candidates[0].content.parts:
                        # Check for inline_data (bytes)
                        if hasattr(part, 'inline_data') and part.inline_data:
                            audio_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                            break
                        # Check for raw data attribute
                        elif hasattr(part, 'data') and part.data:
                            audio_base64 = base64.b64encode(part.data).decode('utf-8')
                            break
                
                if not audio_base64:
                    print("⚠️ Model responded but no audio data was found in parts.")

            except Exception as tts_err:
                # Log error but don't crash the whole response
                print(f"⚠️ TTS Technical Error: {tts_err}")
                # audio_base64 remains empty string, frontend handles it gracefully

        # Return the multi-modal response
        return jsonify({
            "answer": answer, 
            "transcription": user_input, 
            "audio": audio_base64
        })

    except Exception as e:
        print(f"🛑 Critical Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 SmartHIVCare V3.0 (2026 Stable) is LIVE")
    print("🔗 Local: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
## 🏥 SmartHIVCare: AI-Powered Medical Educational Assistant
- This is a chatbot developed by RAG-LLM. This chatboat is create for HIV patients

SmartHIVCare is a bimodal (Voice & Text) chatbot designed for clinical educational support. It utilizes RAG (Retrieval-Augmented Generation) to provide accurate answers based on specific medical PDF documentation, supporting both English and Amharic languages. 

🌟 Key Features

- Bilingual Support: Processes and responds in English and Amharic. 


- Voice Integration: * STT (Speech-to-Text): Uses the OpenAI Whisper-medium model for high-accuracy transcription of Amharic and English audio. 


- TTS (Text-to-Speech): Provides vocalized responses via gTTS. 

- Medical RAG Pipeline: * Indexes PDF medical records into a Chroma DB vector database.

- Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for cross-lingual embedding alignment.

- LLM Intelligence: Powered by Gemini 2.5 Flash for supportive, conversational reasoning.


- Secure Infrastructure: Features a PostgreSQL-backed user authentication system with email verification.

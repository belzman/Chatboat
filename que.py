import os
import re
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.helper import preprocess_text  # <-- FIXED: Import added to balance ingest logic

# 1. Load environment variables and disable telemetry to prevent capture() exceptions
load_dotenv()
os.environ["ANONYMOUS_TELEMETRY"] = "False"

def generate_answer(user_query, language="English"):
    """
    RAG Logic: Preprocesses raw text input, converts query to vectors, 
    searches ChromaDB, and generates a clinical response using Gemini.
    """
    try:
        # 2. Setup Multilingual Embeddings (Identical configuration to ingest.py)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 3. Load the Vector Database safely
        persist_dir = "./chroma_db"
        if not os.path.exists(persist_dir):
            print("❌ DEBUG ERROR: 'chroma_db' directory not found. Did you run ingest.py?")
            return "Error: Vector database not found. Please run ingest.py first."
        
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

        # 4. Retrieve relevant chunks (Top 4 matches)
        # --- FIXED: Preprocess the incoming user query so it matches DB vector signatures ---
        cleaned_query = preprocess_text(user_query)
        docs = db.similarity_search(cleaned_query, k=4)
        
        if not docs:
            return "ተገቢ መረጃ አልተገኘም (No relevant medical info found in the database records)."
            
        # Compile context strings and track document sources
        context_text = ""
        sources = set()
        for doc in docs:
            context_text += f"{doc.page_content}\n\n"
            source_path = doc.metadata.get('source', 'Unknown')
            sources.add(os.path.basename(source_path))

        # 5. Initialize LLM (Gemini 3.1 Flash Lite)
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview", 
            temperature=0.1
        )
        # 6. Prompt Template for Clinical Accuracy
        template = """You are a supportive Medical Educational Assistant for HIV care. 
        Answer the question accurately using ONLY the provided context.
        Provide a clear, conversational response without using bold symbols (*) or hashtags (#).
        Respond in {lang}.
        Context: {ctx}
        Question: {q}
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        
        # 7. Execution with automated retry for rate limits
        for attempt in range(2): 
            try:
                response = chain.invoke({
                    "ctx": context_text, 
                    "q": cleaned_query, 
                    "lang": language
                })
                
                # Extract text content dynamically based on response types
                if hasattr(response, 'content'):
                    res_val = response.content
                else:
                    res_val = response

                if isinstance(res_val, dict):
                    answer_text = res_val.get('text', str(res_val))
                elif isinstance(res_val, list):
                    answer_text = " ".join([part.get('text', '') if isinstance(part, dict) else str(part) for part in res_val])
                else:
                    answer_text = str(res_val)

                # Emergency JSON structural parsing if API defaults to raw string format
                if "{'type': 'text'" in answer_text:
                    match = re.search(r"'text':\s*'(.*?)',", answer_text, re.DOTALL)
                    if match:
                        answer_text = match.group(1)

                # Format Sentences and clean markdown tags
                clean_answer = answer_text.replace("*", "").replace("#", "").strip()
                
                # Split cleanly by standard English endpoints or Amharic (።)
                sentences = re.split(r'(?<=[።\.!?])\s+', clean_answer)
                formatted_answer = "\n\n".join([s.strip() for s in sentences if s.strip()])
                
                source_list = ", ".join(sources) if sources else "Unknown Source"
                return f"{formatted_answer}\n\n📚 የመረጃ ምንጭ (Sources): {source_list}"
            
            except Exception as e:
                if "429" in str(e):
                    print(f"⚠️ Rate limit hit. Retrying in 3 seconds... (Attempt {attempt + 1})")
                    time.sleep(3) 
                else:
                    raise e

    except Exception as e:
        print(f"❌ CRITICAL ERROR IN que.py: {str(e)}")
        return f"System error during answer generation: {str(e)}"
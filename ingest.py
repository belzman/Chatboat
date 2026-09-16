import os
import time
import shutil
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from src.helper import preprocess_text

# Silence the legacy cryptography / ARC4 warnings from pypdf
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load environment variables and disable Chroma telemetry globally
load_dotenv()
os.environ["ANONYMOUS_TELEMETRY"] = "False"

def ingest_docs():
    persist_dir = "./chroma_db"
    knowledge_base_dir = './knowledge_base'
    
    # Robust Windows directory removal with a fall-back sleep timer
    if os.path.exists(persist_dir):
        print(f"🗑️ Removing existing database at {persist_dir}...")
        try:
            shutil.rmtree(persist_dir)
        except PermissionError:
            print("⚠️ Database file is locked by another process. Waiting 3 seconds to retry...")
            time.sleep(3)
            try:
                shutil.rmtree(persist_dir)
            except Exception as e:
                print(f"❌ Could not clear database automatically: {e}")
                print("👉 Please manually delete the './chroma_db' folder or close active python processes.")
                return

    # 2. Initialize Validated Multilingual Embeddings LOCALLY
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}  # Change to 'cuda' if a GPU is available
    )

    # 3. Medical-Grade Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "።", " ", ""]
    )

    # Initialize New Chroma DB
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    if not os.path.exists(knowledge_base_dir):
        print(f"❌ ERROR: Knowledge base directory '{knowledge_base_dir}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(knowledge_base_dir) if f.endswith('.pdf')]
    print(f"📂 Found {len(pdf_files)} PDFs in {knowledge_base_dir}.")

    total_chunks = 0

    for index, filename in enumerate(pdf_files):
        file_path = os.path.join(knowledge_base_dir, filename)
        print(f"📄 [{index+1}/{len(pdf_files)}] Processing: {filename}")

        try:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            
            # Preprocess Amharic/English text using your src/helper.py logic
            for doc in raw_docs:
                doc.page_content = preprocess_text(doc.page_content)
            
            chunks = text_splitter.split_documents(raw_docs)
            file_chunk_count = len(chunks)
            total_chunks += file_chunk_count
            print(f"  ✂️ Created {file_chunk_count} chunks for this file.")

            # Batch Upload logic
            batch_size = 20  
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                success = False
                while not success:
                    try:
                        db.add_documents(batch)
                        success = True
                    except Exception as e:
                        if "429" in str(e):
                            print(f"  ⏳ Rate limit reached. Waiting 60s...")
                            time.sleep(60)
                        else:
                            raise e 
            
            print(f"  ✅ {filename} fully indexed.")

        except Exception as e:
            print(f"  ❌ Failed to process {filename}: {e}")
            continue 

    print("\n" + "="*30)
    print(f"🚀 INGESTION COMPLETE")
    print(f"📊 Total Chunks Indexed: {total_chunks}")
    print("="*30)

if __name__ == "__main__":
    ingest_docs()
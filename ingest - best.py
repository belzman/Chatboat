import os
import time
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from src.helper import preprocess_text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def ingest_docs():
    persist_dir = "./chroma_db"
    knowledge_base_dir = './knowledge_base'
    
    # NEW: Remove existing database to replace with new files
    if os.path.exists(persist_dir):
        print(f"🗑️ Removing existing database at {persist_dir}...")
        shutil.rmtree(persist_dir)
        time.sleep(1) # Short pause for OS to clear handles

    # 1. Initialize Validated Multilingual Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # 2. Medical-Grade Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Initialize New Chroma DB
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    pdf_files = [f for f in os.listdir(knowledge_base_dir) if f.endswith('.pdf')]
    print(f"📂 Found {len(pdf_files)} PDFs in {knowledge_base_dir}.")

    total_chunks = 0

    for index, filename in enumerate(pdf_files):
        file_path = os.path.join(knowledge_base_dir, filename)
        print(f"📄 [{index+1}/{len(pdf_files)}] Processing: {filename}")

        try:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            
            # Preprocess Amharic/English text
            for doc in raw_docs:
                doc.page_content = preprocess_text(doc.page_content)
            
            chunks = text_splitter.split_documents(raw_docs)
            file_chunk_count = len(chunks)
            total_chunks += file_chunk_count
            print(f"  ✂️ Created {file_chunk_count} chunks for this file.")

            # Batch Upload logic
            batch_size = 5
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
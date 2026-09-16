import os
import time
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
    progress_file = "processed_files.txt"

    # 1. Initialize Validated Multilingual Embeddings
    # This model handles Amharic (🇪🇹) and English (🇬🇧) efficiently
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

    # Initialize Chroma DB
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Track progress
    if not os.path.exists(progress_file):
        open(progress_file, 'w').close()
    
    with open(progress_file, 'r') as f:
        done_files = set(f.read().splitlines())

    pdf_files = [f for f in os.listdir(knowledge_base_dir) if f.endswith('.pdf')]
    print(f"📂 Found {len(pdf_files)} PDFs. {len(done_files)} already indexed.")

    for index, filename in enumerate(pdf_files):
        if filename in done_files:
            print(f"⏭️ Skipping {filename} (Already in Database)")
            continue

        file_path = os.path.join(knowledge_base_dir, filename)
        print(f"📄 [{index+1}/{len(pdf_files)}] Processing: {filename}")

        try:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            
            # Preprocess Amharic/English text
            for doc in raw_docs:
                doc.page_content = preprocess_text(doc.page_content)
            
            chunks = text_splitter.split_documents(raw_docs)
            print(f"  ✂️ Created {len(chunks)} chunks.")

            # Batch Upload logic to avoid API timeouts
            batch_size = 5
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                success = False
                while not success:
                    try:
                        db.add_documents(batch)
                        success = True
                        time.sleep(1) # Short delay for stability
                    except Exception as e:
                        if "429" in str(e):
                            print(f"  ⏳ Rate limit reached. Waiting 60s...")
                            time.sleep(60)
                        else:
                            print(f"  ❌ Batch Error: {e}")
                            raise e 
            
            # Log completed file
            with open(progress_file, 'a') as f:
                f.write(filename + "\n")
            
            print(f"  ✅ {filename} fully indexed.")
            time.sleep(2) 

        except Exception as e:
            print(f"  ❌ Failed to process {filename}: {e}")
            break 

    print(f"🚀 INGESTION COMPLETE: Your medical database is ready!")

if __name__ == "__main__":
    ingest_docs()
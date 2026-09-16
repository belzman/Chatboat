import os
import shutil
import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from src.helper import preprocess_text
from dotenv import load_dotenv

load_dotenv()

def ingest_docs():
    persist_dir = "./chroma_db"
    knowledge_base_dir = './knowledge_base'

    if os.path.exists(persist_dir):
        print(f"🧹 Deleting old database...")
        shutil.rmtree(persist_dir)

    print(f"📂 Loading PDFs...")
    loader = DirectoryLoader(knowledge_base_dir, glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    
    for doc in raw_docs:
        doc.page_content = preprocess_text(doc.page_content)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✂️ Created {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="retrieval_document"
    )

    print("💾 Creating embeddings with Batch Rate-Limiting...")
    
    # Process in batches to avoid 429 Error
    batch_size = 5 
    db = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"📦 Processing batch {i//batch_size + 1}/{(len(chunks)//batch_size)+1}...")
        
        if db is None:
            db = Chroma.from_documents(batch, embeddings, persist_directory=persist_dir)
        else:
            db.add_documents(batch)
        
        # Pause to let the API quota reset
        time.sleep(10) 

    print(f"🚀 SUCCESS: Database indexed without crashing.")

if __name__ == "__main__":
    ingest_docs()
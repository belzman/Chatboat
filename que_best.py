import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables
load_dotenv()

def generate_answer(user_query, language="English"):
    try:
        # 2. Setup Multilingual Embeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        # 3. Load the Vector Database
        persist_dir = "./chroma_db"
        if not os.path.exists(persist_dir):
            return "Error: Vector database not found. Please run ingest.py first."

        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        # 4. Retrieve relevant chunks
        docs = db.similarity_search(user_query, k=4)
        
        if not docs:
            return "ተገቢ መረጃ አልተገኘም። (No relevant medical info found in records.)"

        context_text = ""
        sources = set()
        for doc in docs:
            context_text += f"{doc.page_content}\n\n"
            source_path = doc.metadata.get('source', 'Unknown')
            source_name = os.path.basename(source_path)
            sources.add(source_name)

        # 5. Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1, 
        )
        
        # 6. Prompt Template
        template = """You are a supportive Medical Educational Assistant. 
        Answer the question accurately using ONLY the provided context.
        Provide a clear, conversational response without using bold symbols (*) or hashtags (#).
        
        Respond in {lang}.
        
        Context: {ctx}
        Question: {q}
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm

        # 7. Execution
        for attempt in range(2): 
            try:
                response = chain.invoke({
                    "ctx": context_text, 
                    "q": user_query, 
                    "lang": language
                })
                
                source_list = ", ".join(sources)
                clean_answer = response.content.replace("*", "").replace("#", "")
                
                return f"{clean_answer}\n\n📚 የመረጃ ምንጭ (Sources): {source_list}"
            
            except Exception as e:
                if "429" in str(e):
                    time.sleep(3) 
                else:
                    raise e

    except Exception as e:
        print(f"Error in que.py: {e}")
        return None
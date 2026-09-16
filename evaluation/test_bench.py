import sys
import os
import time

# Path fix to allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from que import generate_answer
except ImportError:
    from que import generate_answer

def run_performance_test():
    print("🧪 PERFORMANCE BENCHMARK: Measuring Bimodal Latency (N=10)\n")
    print("Clinical Target: Response Latency < 4.0 seconds\n")
    print("-" * 50)
    
    # 10 Stress-test queries (Bilingual)
    queries = [
        # English Set
        "How often should I take my ART medication?",
        "What are the common side effects of HIV drugs?",
        "Can I take ART during pregnancy?",
        "What happens if I skip my medication for two days?",
        "Does ART cure HIV completely?",
        
        # Amharic Set
        "መድሃኒቱን መውሰድ ብረሳ ምን ማድረግ አለብኝ?", # What if I forget?
        "የአርቲ መድሃኒት የጎንዮሽ ጉዳቱ ምንድነው?", # Side effects?
        "መድሃኒቱን በባዶ ሆድ መውሰድ ይቻላል?", # Take on empty stomach?
        "የቫይረስ መጠን ምርመራ ለምን ያስፈልጋል?", # Why viral load test?
        "ኤችአይቪ ያለባት እናት ጤናማ ልጅ መውለድ ትችላለች?" # HIV+ mother healthy child?
    ]
    
    latencies = []
    
    for i, q in enumerate(queries, 1):
        print(f"[{i}/10] Testing Query: {q}")
        
        start_time = time.time()
        # This triggers the full RAG pipeline (Retrieval + LLM Generation)
        response = generate_answer(q) 
        latency = time.time() - start_time
        
        latencies.append(latency)
        
        # Clinical standard for real-time digital health is < 4.0s
        status = "✅ PASS" if latency < 4.0 else "⚠️ SLOW"
        print(f"⏱️ Latency: {latency:.2f}s | Status: {status}")
        print("-" * 30)

    # Scientific Summary for Manuscript Table 3
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print("\n📊 MANUSCRIPT DATA SUMMARY:")
    print(f"Average System Latency: {avg_latency:.2f}s")
    print(f"Maximum Latency: {max_latency:.2f}s")
    print(f"Minimum Latency: {min_latency:.2f}s")
    print(f"Success Rate (<4s): {(len([l for l in latencies if l < 4]) / 10) * 100}%")

if __name__ == "__main__":
    run_performance_test()
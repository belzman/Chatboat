import sys
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

# Path fix to allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from que import generate_answer
    from evaluation.eval_metrics import ResearchEvaluator
except ImportError:
    from que import generate_answer
    from eval_metrics import ResearchEvaluator

def run_q1_benchmark():
    evaluator = ResearchEvaluator()
    
    # Ragas metric objects
    faithfulness = Faithfulness()
    answer_relevancy = AnswerRelevancy()
    
    # Expanded Golden Dataset (20 pairs)
    test_set = [
        # --- ENGLISH CLINICAL QUERIES ---
        {"question": "What is the primary goal of Antiretroviral Therapy (ART)?", "ground_truth": "The primary goal is to achieve viral suppression, restore immune function, and reduce HIV-related morbidity and mortality."},
        {"question": "How often should a patient undergo viral load testing?", "ground_truth": "Typically, viral load is tested 6 months after starting ART, and then every 12 months if the patient is stable."},
        {"question": "What should I do if I miss a dose of my ART medication?", "ground_truth": "Take the missed dose as soon as you remember, unless it is almost time for your next dose. Do not double the dose."},
        {"question": "What are the common side effects of Efavirenz?", "ground_truth": "Common side effects include dizziness, insomnia, and vivid dreams, usually occurring during the first few weeks."},
        {"question": "Can ART cure HIV completely?", "ground_truth": "No, ART is not a cure. It suppresses the virus to undetectable levels but must be taken for life."},
        {"question": "What is a 'U=U' status in HIV care?", "ground_truth": "'Undetectable = Untransmissible' means a person with an undetectable viral load cannot transmit HIV sexually."},
        {"question": "Why is adherence so important for ART success?", "ground_truth": "High adherence prevents drug resistance and ensures the virus remains suppressed."},
        {"question": "What is the role of CD4 cells in HIV?", "ground_truth": "CD4 cells are white blood cells that fight infection; HIV attacks these cells, weakening the immune system."},
        {"question": "Can a pregnant woman on ART prevent transmission to her baby?", "ground_truth": "Yes, with effective ART and viral suppression, the risk of mother-to-child transmission is less than 1%."},
        {"question": "What is Post-Exposure Prophylaxis (PEP)?", "ground_truth": "PEP is short-term medication taken within 72 hours of a possible HIV exposure to prevent infection."},

        # --- AMHARIC CLINICAL QUERIES ---
        {"question": "የአርቲ (ART) መድሃኒት ዋና ጥቅሙ ምንድነው?", "ground_truth": "ዋናው ጥቅሙ በደም ውስጥ ያለውን የቫይረስ መጠን መቀነስ እና በሽታ የመከላከል አቅምን ማጠናከር ነው።"},
        {"question": "መድሃኒቱን መውሰድ ብረሳ ምን ማድረግ አለብኝ?", "ground_truth": "ያስታወሱበት ሰዓት ከሚቀጥለው የመድሃኒት ሰዓት ጋር ካልተቀራረበ ወዲያውኑ ይውሰዱ፤ ነገር ግን ሁለት መጠን በአንድ ላይ አይውሰዱ።"},
        {"question": "መድሃኒቱን ስጀምር ምን አይነት የጎንዮሽ ጉዳቶች ሊያጋጥሙኝ ይችላሉ?", "ground_truth": "ራስ ምታት፣ ማቅለሽለሽ ወይም የእንቅልፍ መዛባት ሊያጋጥሙ ይችላሉ፤ እነዚህም አብዛኛውን ጊዜ ከጥቂት ሳምንታት በኋላ ይጠፋሉ።"},
        {"question": "የቫይረስ መጠን (Viral load) ምርመራ በስንት ጊዜ መደረግ አለበት?", "ground_truth": "መድሃኒት በጀመሩ በ6ኛው ወር፣ ከዚያም በየአመቱ ምርመራው መደረግ አለበት።"},
        {"question": "አርቲ (ART) ኤችአይቪን ሙሉ በሙሉ ያድናል?", "ground_truth": "አይ፣ መድሃኒቱ ቫይረሱን ያዳክማል እንጂ ሙሉ በሙሉ አያድንም። ስለዚህ እድሜ ልክ መወሰድ አለበት።"},
        {"question": "የሲዲ 4 (CD4) ሴል መውረድ ምን ማለት ነው?", "ground_truth": "የሲዲ 4 ሴል መውረድ የሰውነት በሽታ የመከላከል አቅም መዳከሙን እና ለበሽታ የመጋለጥ እድል መጨመሩን ያሳያል።"},
        {"question": "መድሃኒት በአግባቡ አለመውሰድ ምን ጉዳት አለው?", "ground_truth": "መድሃኒት በአግባቡ አለመውሰድ ቫይረሱ መድሃኒቱን እንዲለምድ (Resistance) እና ህክምናው እንዳይሰራ ያደርጋል።"},
        {"question": "ኤችአይቪ ያለባት እናት ጤናማ ልጅ መውለድ ትችላለች?", "ground_truth": "አዎ፣ እናትየው መድሃኒቱን በአግባቡ ከወሰደች እና የቫይረስ መጠኑ ከቀነሰ ጤናማ ልጅ መውለድ ትችላለች።"},
        {"question": "የኤችአይቪ መድሃኒት ከምግብ ጋር ነው የሚወሰደው?", "ground_truth": "እንደ መድሃኒቱ አይነት ይለያያል፤ አንዳንዶቹ ከምግብ ጋር አንዳንዶቹ ደግሞ በባዶ ሆድ ይወሰዳሉ።"},
        {"question": "የቫይረስ መጠኑ የማይታይ (Undetectable) መሆን ምን ማለት ነው?", "ground_truth": "ይህ ማለት በደም ውስጥ ያለው የቫይረስ መጠን በጣም አነስተኛ ስለሆነ በምርመራ አይታይም እና ወደ ሌላ ሰው የመተላለፍ እድሉ የለም ማለት ነው።"}
    ]

    results_data = []

    print("\n🚀 Starting Q1 Comprehensive Benchmarking (N=20)...")
    
    for item in test_set:
        print(f"📝 Testing: {item['question'][:40]}...")
        answer = generate_answer(item['question'])
        
        # Calculate BERTScore and BLEU
        ling_scores = evaluator.compute_linguistic_quality([answer], [[item['ground_truth']]])
        
        results_data.append({
            "Question": item['question'],
            "Prediction": answer,
            "Ground_Truth": item['ground_truth'],
            "BERTScore": ling_scores['BERTScore_F1'],
            "BLEU": ling_scores['BLEU']
        })

    # Save Results
    df = pd.DataFrame(results_data)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "q1_comprehensive_results.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Benchmarking Complete. Data saved to {output_path}")

if __name__ == "__main__":
    run_q1_benchmark()
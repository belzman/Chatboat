import os
import time
import pandas as pd
import re
import warnings
from dotenv import load_dotenv
from datasets import Dataset
import evaluate as hf_evaluate

# CORE COMPONENTS
from que import generate_answer  # Ensure que.py is in the same directory
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
load_dotenv()

# 1. INITIALIZE EVALUATION ENGINES
print("📂 Initializing Lexical (BLEU) and Semantic (BERTScore) Engines...")
bleu = hf_evaluate.load("bleu")
bertscore = hf_evaluate.load("bertscore")

# 2. THE CLINICAL DATASET (50 QUESTIONS)
eval_data = [
    # --- English: Treatment, Prevention, & Knowledge (25) ---
    {"question": "What is the primary goal of Antiretroviral Therapy (ART)?", "ground_truth": "The primary goal is to achieve viral suppression, restore immune function, and reduce HIV-related morbidity and mortality.", "language": "English"},
    {"question": "How often should a patient undergo viral load testing?", "ground_truth": "Typically, viral load is tested 6 months after starting ART, and then every 12 months if the patient is stable.", "language": "English"},
    {"question": "What should I do if I miss a dose of my ART medication?", "ground_truth": "Take the missed dose as soon as you remember, unless it is almost time for your next dose. Do not double the dose.", "language": "English"},
    {"question": "Can I stop taking ART if my viral load is undetectable?", "ground_truth": "No, ART is a lifelong treatment. Stopping it will allow the virus to multiply again.", "language": "English"},
    {"question": "What is the importance of 95% adherence to ART?", "ground_truth": "High adherence is necessary to prevent drug resistance and maintain viral suppression.", "language": "English"},
    {"question": "Does ART cure HIV?", "ground_truth": "No, ART does not cure HIV, but it controls the virus so you can live a long, healthy life.", "language": "English"},
    {"question": "Can I share my ART medication with someone else who has HIV?", "ground_truth": "No, HIV treatment is individualized based on specific medical needs and resistance profiles.", "language": "English"},
    {"question": "What does U=U mean?", "ground_truth": "Undetectable = Untransmittable. It means a person with an undetectable viral load cannot transmit HIV through sex.", "language": "English"},
    {"question": "What is PrEP?", "ground_truth": "Pre-Exposure Prophylaxis is a daily pill taken by HIV-negative people to prevent infection.", "language": "English"},
    {"question": "What is PEP?", "ground_truth": "Post-Exposure Prophylaxis is emergency medication taken within 72 hours of possible HIV exposure.", "language": "English"},
    {"question": "Can HIV be transmitted through breastfeeding?", "ground_truth": "Yes, but the risk is significantly reduced if the mother is on effective ART with an undetectable viral load.", "language": "English"},
    {"question": "How can a pregnant woman prevent passing HIV to her baby?", "ground_truth": "By taking ART consistently during pregnancy, delivery, and breastfeeding.", "language": "English"},
    {"question": "Does a condom prevent HIV transmission?", "ground_truth": "Yes, consistent and correct use of condoms highly reduces the risk of HIV transmission.", "language": "English"},
    {"question": "What are common side effects of starting ART?", "ground_truth": "Common side effects include nausea, fatigue, headache, and dizziness, which usually fade after a few weeks.", "language": "English"},
    {"question": "What should I do if my ART causes severe skin rashes?", "ground_truth": "Contact your healthcare provider immediately, as this could be a serious allergic reaction.", "language": "English"},
    {"question": "Can I take herbal medicine with my ART?", "ground_truth": "You should consult your doctor first, as some herbs can interfere with the effectiveness of ART.", "language": "English"},
    {"question": "Is alcohol safe to consume while on ART?", "ground_truth": "Moderate alcohol use is usually okay, but excessive drinking can lead to missed doses and liver strain.", "language": "English"},
    {"question": "What kind of diet is recommended for people living with HIV?", "ground_truth": "A balanced diet rich in fruits, vegetables, lean protein, and whole grains is recommended.", "language": "English"},
    {"question": "What is the difference between HIV and AIDS?", "ground_truth": "HIV is the virus that causes the infection; AIDS is the advanced stage where the immune system is severely damaged.", "language": "English"},
    {"question": "How is HIV diagnosed?", "ground_truth": "HIV is diagnosed through blood or oral fluid tests that detect antibodies and antigens.", "language": "English"},
    {"question": "Is there a vaccine for HIV?", "ground_truth": "Currently, there is no effective vaccine for HIV, though research is ongoing.", "language": "English"},
    {"question": "How soon after exposure should I get tested?", "ground_truth": "Most tests can detect HIV within 23 to 90 days after exposure.", "language": "English"},
    {"question": "Can mosquitoes transmit HIV?", "ground_truth": "No, HIV cannot be transmitted by mosquitoes or other insects.", "language": "English"},
    {"question": "What is a CD4 count?", "ground_truth": "A CD4 count measures the number of T-cells in your blood, which indicates the strength of your immune system.", "language": "English"},
    {"question": "Is HIV a death sentence?", "ground_truth": "No, with modern ART, people living with HIV can have a normal life expectancy.", "language": "English"},

    # --- Amharic: Treatment, Prevention, & Knowledge (25) ---
    {"question": "የአርቲ (ART) መድሃኒት ዋና ጥቅሙ ምንድነው?", "ground_truth": "ዋናው ጥቅሙ በደም ውስጥ ያለውን የቫይረስ መጠን መቀነስ እና በሽታ የመከላከል አቅምን ማጠናከር ነው።", "language": "Amharic"},
    {"question": "መድሃኒቱን መውሰድ ብረሳ ምን ማድረግ አለብኝ?", "ground_truth": "ያስታወሱበት ሰዓት ከሚቀጥለው የመድሃኒት ሰዓት ጋር ካልተቀራረበ ወዲያውኑ ይውሰዱ፤ ነገር ግን ሁለት መጠን በአንድ ላይ አይውሰዱ።", "language": "Amharic"},
    {"question": "የቫይረስ መጠን ምርመራ (Viral Load) በስንት ጊዜ መደረግ አለበት?", "ground_truth": "መድሃኒት እንደተጀመረ በ6 ወር፣ ከዚያም በየአመቱ ምርመራው መደረግ አለበት።", "language": "Amharic"},
    {"question": "ጤነኛ ሆኖ ከተሰማኝ መድሃኒት ማቆም እችላለሁ?", "ground_truth": "አይቻልም፤ መድሃኒት ማቆም ቫይረሱ እንደገና እንዲበረታ እና መድሃኒቱን እንዲላመድ ያደርጋል።", "language": "Amharic"},
    {"question": "መድሃኒቱን ሁልጊዜ በተመሳሳይ ሰዓት መውሰድ ለምን ያስፈልጋል?", "ground_truth": "በደም ውስጥ ያለውን የመድሃኒት መጠን ወጥ በሆነ መልኩ ለመጠበቅ እና ቫይረሱ እንዳያገግም ለመከላከል ነው።", "language": "Amharic"},
    {"question": "የኤአርቲ መድሃኒት ቫይረሱን ሙሉ በሙሉ ያጠፋል?", "ground_truth": "አያጠፋም፤ ነገር ግን ቫይረሱ እንዳይባዛ በማድረግ ረጅም እና ጤናማ ህይወት ለመምራት ይረዳል።", "language": "Amharic"},
    {"question": "መድሃኒቴን ለሌላ ሰው ማካፈል እችላለሁ?", "ground_truth": "አይቻልም፤ መድሃኒት የሚታዘዘው በግለሰቡ የጤና ሁኔታ እና የቫይረስ አይነት ላይ ተመስርቶ ነው።", "language": "Amharic"},
    {"question": "U=U ማለት ምን ማለት ነው?", "ground_truth": "በደም ውስጥ ያለው የቫይረስ መጠን በጣም ዝቅተኛ ከሆነ (undetectable) ቫይረሱ በግብረ ስጋ ግንኙነት አይተላለፍም (untransmittable) ማለት ነው።", "language": "Amharic"},
    {"question": "PrEP (ፕሪፕ) ምንድነው?", "ground_truth": "ቫይረሱ የሌለባቸው ሰዎች ግን ለቫይረሱ ተጋላጭ ከሆኑ አስቀድመው በመውሰድ ራሳቸውን የሚከላከሉበት መድሃኒት ነው።", "language": "Amharic"},
    {"question": "PEP (ፔፕ) ምንድነው?", "ground_truth": "ለቫይረሱ ተጋላጭ ከሆኑ በኋላ በ72 ሰአታት ውስጥ በመጀመር ስርጭቱን ለመከላከል የሚወሰድ አስቸኳይ መድሃኒት ነው።", "language": "Amharic"},
    {"question": "ኤች አይ ቪ በጡት ወተት ይተላለፋል?", "ground_truth": "አዎ ይተላለፋል፤ ነገር ግን እናቲቱ መድሃኒቷን በትክክል የምትወስድ ከሆነ የመተላለፍ እድሉ በጣም ዝቅተኛ ነው።", "language": "Amharic"},
    {"question": "አንዲት ነፍሰ ጡር እናት ቫይረሱ ወደ ልጇ እንዳይተላለፍ ምን ማድረግ አለባት?", "ground_truth": "የኤአርቲ መድሃኒቷን በእርግዝና፣ በወሊድ እና ጡት በማጥባት ጊዜ በትክክል መውሰድ አለባት።", "language": "Amharic"},
    {"question": "ኮንዶም ኤች አይ ቪን ይከላከላል?", "ground_truth": "አዎ፣ ኮንዶምን በትክክል እና ሁልጊዜ መጠቀም ኤች አይ ቪ የመያዝ እድልን በከፍተኛ ሁኔታ ይቀንሳል።", "language": "Amharic"},
    {"question": "መድሃኒት ሲጀመር ምን አይነት የጎንዮሽ ጉዳቶች ሊያጋጥሙ ይችላሉ?", "ground_truth": "ማቅለሽለሽ፣ ራስ ምታት፣ ድካም እና መደባለቅ ሊታዩ ይችላሉ፤ እነዚህም በጥቂት ሳምንታት ውስጥ ይጠፋሉ።", "language": "Amharic"},
    {"question": "ከባድ ሽፍታ ካጋጠመኝ ምን ማድረግ አለብኝ?", "ground_truth": "ወዲያውኑ ወደ ጤና ተቋም በመሄድ ሀኪም ማማከር ያስፈልጋል።", "language": "Amharic"},
    {"question": "ባህላዊ መድሃኒቶችን ከኤአርቲ ጋር መውሰድ ይቻላል?", "ground_truth": "አይመከርም፤ አንዳንድ ባህላዊ መድሃኒቶች የኤአርቲ መድሃኒቱን ስራ ሊያስተጓጉሉ ይችላሉ።", "language": "Amharic"},
    {"question": "መድሃኒት በሚወሰድበት ጊዜ አልኮል መጠጣት ይቻላል?", "ground_truth": "ከመጠን ያለፈ አልኮል መድሃኒት መርሳትን እና የጉበት ጫናን ስለሚያስከትል አይመከርም።", "language": "Amharic"},
    {"question": "ኤች አይ ቪ ያለባቸው ሰዎች ምን አይነት አመጋገብ ያስፈልጋቸዋል?", "ground_truth": "ፍራፍሬ፣ አትክልት፣ ፕሮቲን እና ጥራጥሬዎችን የያዘ የተመጣጠነ ምግብ መመገብ ይመከራል።", "language": "Amharic"},
    {"question": "በኤች አይ ቪ እና በኤድስ መካከል ያለው ልዩነት ምንድነው?", "ground_truth": "ኤች አይ ቪ ቫይረሱ ነው፤ ኤድስ ደግሞ በቫይረሱ ምክንያት የመከላከል አቅም በጣም የሚዳከምበት ደረጃ ነው።", "language": "Amharic"},
    {"question": "ኤች አይ ቪ እንዴት ይታወቃል?", "ground_truth": "በደም ምርመራ ወይም በአፍ ውስጥ በሚወሰድ ፈሳሽ አማካኝነት በሚደረግ የምርመራ አይነት ይታወቃል።", "language": "Amharic"},
    {"question": "ለኤች አይ ቪ ክትባት አለ?", "ground_truth": "እስካሁን ድረስ ውጤታማ የሆነ የኤች አይ ቪ ክትባት የለም።", "language": "Amharic"},
    {"question": "ቫይረሱ ካለበት ሰው ጋር ንክኪ ቢኖረኝ መቼ መመርመር አለኝ?", "ground_truth": "አብዛኛውን ጊዜ ከ23 እስከ 90 ቀናት ባለው ጊዜ ውስጥ ምርመራው ውጤት ያሳያል።", "language": "Amharic"},
    {"question": "ትንኝ ኤች አይ ቪን ታስተላልፋለች?", "ground_truth": "አይ፣ ኤች አይ ቪ በትንኝ ወይም በሌሎች ነፍሳት አይተላለፍም።", "language": "Amharic"},
    {"question": "የሲዲ 4 (CD4) መጠን ምንድነው?", "ground_truth": "በደም ውስጥ ያሉ ነጭ የደም ሴሎችን በመቁጠር የሰውነትን የመከላከል አቅም የሚለካ ምርመራ ነው።", "language": "Amharic"},
    {"question": "ኤች አይ ቪ መያዝ ማለት ሞት ማለት ነው?", "ground_truth": "አይደለም፤ ዘመናዊ የኤአርቲ መድሃኒቶችን በመጠቀም እንደ ማንኛውም ሰው ረጅም እድሜ መኖር ይቻላል።", "language": "Amharic"}
]

def run_evaluation():
    results = []
    print(f"🚀 Starting Automated Evaluation for SmartHIVCare ({len(eval_data)} queries)...")

    # Use Gemini 1.5 Pro as the judge for RAGAS metrics
    critic_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    for i, entry in enumerate(eval_data):
        start_time = time.time()
        
        try:
            # Query the RAG system
            system_response = generate_answer(entry['question'], language=entry['language'])
            
            # Extract clinical answer and sources from your formatted output
            clean_answer = system_response.split("📚")[0].strip()
            source_match = re.search(r"Sources\): (.*)", system_response)
            contexts = [source_match.group(1)] if source_match else ["Clinical Guideline Reference"]
            
        except Exception as e:
            print(f"❌ Error during generation at index {i}: {e}")
            clean_answer, contexts = "Generation Error", ["None"]

        latency = time.time() - start_time
        
        results.append({
            "question": entry['question'],
            "answer": clean_answer,
            "ground_truth": entry['ground_truth'],
            "contexts": contexts,
            "latency": latency,
            "language": entry['language']
        })
        
        print(f"✅ [{i+1}/{len(eval_data)}] {entry['language']} Query Processed ({latency:.2f}s)")

    # 3. METRICS CALCULATION
    df = pd.DataFrame(results)
    
    # BLEU Score
    bleu_score = bleu.compute(predictions=df['answer'].tolist(), 
                              references=[[gt] for gt in df['ground_truth'].tolist()])
    
    # BERTScore using Multilingual Model for Amharic compatibility
    bert_res = bertscore.compute(predictions=df['answer'].tolist(), 
                                 references=df['ground_truth'].tolist(), 
                                 model_type="bert-base-multilingual-cased")

    # RAGAS (Faithfulness and Relevancy)
    print("\n🤖 Running RAGAS Clinical Alignment Checks (LLM-as-a-judge)...")
    ds = Dataset.from_pandas(df)
    try:
        ragas_res = evaluate(ds, metrics=[faithfulness, answer_relevancy], llm=critic_llm)
    except Exception as e:
        ragas_res = f"Ragas Score Calculation Failed: {e}"

    # 4. FINAL REPORT GENERATION
    print("\n" + "="*70)
    print("📈 SMART HIV CARE: BILINGUAL RAG EVALUATION REPORT")
    print(f"Evaluation Date:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Queries:     {len(eval_data)} (25 Eng / 25 Amh)")
    print(f"Average Latency:   {df['latency'].mean():.2f} seconds")
    print(f"Overall BLEU:      {bleu_score['bleu']:.4f}")
    print(f"Overall BERTScore: {sum(bert_res['f1'])/len(bert_res['f1']):.4f}")
    print("-" * 70)
    print("RAGAS EVALUATION (0.0 to 1.0):")
    print(ragas_res)
    print("="*70)

    # Save to CSV for Publication Appendix
    df.to_csv("smarthivcare_eval_final.csv", index=False, encoding='utf-8-sig')
    print("💾 Full detailed report saved to 'smarthivcare_eval_final.csv'")

if __name__ == "__main__":
    run_evaluation()
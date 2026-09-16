import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
import evaluate as hf_evaluate
from src.helper import tokenize_to_words  # <-- IMPORT THE SHARED TOKENIZER

warnings.filterwarnings("ignore")
load_dotenv()
os.environ["ANONYMOUS_TELEMETRY"] = "False"

try:
    from que import generate_answer 
# Handle module importing patterns gracefully 
except ImportError:
    print("❌ CRITICAL ERROR: 'que.py' not found in the current directory.")
    exit(1)

# Dataset containing 50 bilingual evaluations
eval_data = [
    # --- English Dataset ---
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

    # --- Amharic Dataset ---
    {"question": "የአርቲ (ART) መድሃኒት ዋና ጥቅሙ ምንድነው?", "ground_truth": "ዋናው ጥቅሙ በደም ውስጥ ያለውን የቫይረስ መጠን መቀነስ እና በሽታ የመከላከል አቅምን ማጠናከር ነው።", "language": "Amharic"},
    {"question": "መድሃኒቱን መውሰድ ብረሳ ምን ማድረግ አለብኝ?", "ground_truth": "ያስታወሱበት ሰዓት ከሚቀጥለው የመድሃኒት ሰዓት ጋር ካልተቀራረበ ወዲያውኑ ይውሰዱ፤ ነገር ግን ሁለት መጠን በአንድ ላይ አይውሰዱ።", "language": "Amharic"},
    {"question": "የአንድ ሰው የቫይረስ መጠን ምርመራ (Viral Load) በስንት ጊዜ መደረግ አለበት?", "ground_truth": "መድሃኒት እንደተጀመረ በ6 ወር፣ ከዚያም በየአመቱ ምርመራው መደረግ አለበት።", "language": "Amharic"},
    {"question": "ጤነኛ ሆኖ ከተሰማኝ መድሃኒት ማቆም እችላለሁ?", "ground_truth": "አይቻልም፤ መድሃኒት ማቆም ቫይረሱ እንደገና እንዲበረታ እና መድሃኒቱን እንዲላመድ ያደርጋል።", "language": "Amharic"},
    {"question": "መድሃኒቱን ሁልጊዜ በተመሳሳይ ሰዓት መውሰድ ለምን ያስፈልጋል?", "ground_truth": "በደም ውስጥ ያለውን የመድሃኒት መጠን ወጥ በሆነ መልኩ ለመጠበቅ እና ቫይረሱ እንዳያገግም ለመከላከል ነው።", "language": "Amharic"},
    {"question": "የኤአርቲ መድሃኒት ቫይረሱን ሙሉ በሙሉ ያጠፋል?", "ground_truth": "አያጠፋም፤ ነገር ግን ቫይረሱ እንዳይባዛ በማድረግ ረጅም እና ጤናማ ህይወት ለመምራት ይረዳል።", "language": "Amharic"},
    {"question": "መድሃኒቴን ለሌላ ሰው ማካፈል እችላለሁ?", "ground_truth": "አይቻልም፤ መድሃኒት የሚታዘዘው በግለሰቡ የጤና ሁኔታ እና የቫይረስ አይነት ላይ ተመስርቶ ነው።", "language": "Amharic"},
    {"question": "U=U ማለት ምን ማለት ነው?", "ground_truth": "በደም ውስጥ ያለው የቫይረስ መጠን በጣም ዝቅተኛ ከሆነ (undetectable) ቫይረሱ በግብረ ስጋ ግንኙነት አይተላለፍም (untransmittable) ማለት ነው።", "language": "Amharic"},
    {"question": "PrEP (ፕሪፕ) ምንድነው?", "ground_truth": "慢性病 ቫይረሱ የሌለባቸው ሰዎች ግን ለቫይረሱ ተጋላጭ ከሆኑ አስቀድመው በመውሰድ ራሳቸውን የሚከላከሉበት መድሃኒት ነው።", "language": "Amharic"},
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

    bleu = hf_evaluate.load("bleu")
    bertscore = hf_evaluate.load("bertscore")
    rouge = hf_evaluate.load("rouge")
    chrf = hf_evaluate.load("chrf")

    for i, entry in enumerate(eval_data):
        start_time = time.time()
        try:
            system_response = generate_answer(entry["question"], language=entry["language"])
            
            if isinstance(system_response, str) and "📚" in system_response:
                clean_answer = system_response.split("📚")[0].strip()
            else:
                clean_answer = str(system_response).strip()
                
        except Exception as e:
            print(f"❌ Error during generation at index {i}: {e}")
            clean_answer = "Generation Error"

        latency = time.time() - start_time
        results.append({
            "question": entry["question"],
            "answer": clean_answer,
            "ground_truth": entry["ground_truth"],
            "latency": latency,
            "language": entry["language"]
        })
        print(f"✅ [{i+1}/{len(eval_data)}] {entry['language']} Processed ({latency:.2f}s)")

    df = pd.DataFrame(results)

    print("\n📊 Computing Word-Wise Lexical Overlay, Sequence Overlap, and Semantic Scores...")
    bleu_scores = []
    bert_scores = []
    rouge_l_scores = []
    chrf_plus_plus_scores = []

    for pred, ref in zip(df["answer"].tolist(), df["ground_truth"].tolist()):
        if pred == "Generation Error":
            bleu_scores.append(0.0)
            bert_scores.append(0.0)
            rouge_l_scores.append(0.0)
            chrf_plus_plus_scores.append(0.0)
            continue
            
        try:
            # --- Convert strings to clean lists of words using helper.py ---
            pred_word_tokens = tokenize_to_words(pred)
            ref_word_tokens = tokenize_to_words(ref)

            # Re-verify that tokens were created to avoid evaluation crashes
            if not pred_word_tokens or not ref_word_tokens:
                bleu_scores.append(0.0)
                rouge_l_scores.append(0.0)
                chrf_plus_plus_scores.append(0.0)
                continue

            # Reconstruct elements back to space-separated values
            pred_input = " ".join(pred_word_tokens)
            ref_input = " ".join(ref_word_tokens)

            # --- BLEU Calculation with Forced Split Tokenizer ---
            # Using tokenizer=lambda x: x.split() locks evaluation purely onto your stemmed words
            bleu_val = bleu.compute(
                predictions=[pred_input], 
                references=[[ref_input]],
                tokenizer=lambda x: x.split(),  
                smooth=True  
            )["bleu"]
            bleu_scores.append(bleu_val)
        except Exception as e:
            print(f"⚠️ BLEU failure bypassed: {e}")
            bleu_scores.append(0.0)

        # --- ROUGE-L Calculation with Forced Split Tokenizer ---
        try:
            rouge_val = rouge.compute(
                predictions=[pred_input],
                references=[ref_input],
                tokenizer=lambda x: x.split(),  
                rouge_types=["rougeL"]
            )["rougeL"]
            rouge_l_scores.append(rouge_val)
        except Exception as e:
            print(f"⚠️ ROUGE-L failure bypassed: {e}")
            rouge_l_scores.append(0.0)

        # --- Compute ChrF++ (Character n-gram F-score + Word n-grams) ---
        try:
            chrf_val = chrf.compute(
                predictions=[pred],
                references=[[ref]],
                word_order=2
            )["score"] / 100.0  
            chrf_plus_plus_scores.append(chrf_val)
        except Exception as e:
            print(f"⚠️ ChrF++ failure bypassed: {e}")
            chrf_plus_plus_scores.append(0.0)

        # --- Compute BERTScore ---
        try:
            bert_val = bertscore.compute(
                predictions=[pred], 
                references=[ref],
                model_type="bert-base-multilingual-cased"
            )["f1"][0]
            bert_scores.append(bert_val)
        except Exception:
            bert_scores.append(0.0)

    # Statistical Formulation
    mean_bleu = np.mean(bleu_scores)
    sd_bleu = np.std(bleu_scores)
    
    mean_bert = np.mean(bert_scores)
    sd_bert = np.std(bert_scores)

    mean_rouge_l = np.mean(rouge_l_scores)
    sd_rouge_l = np.std(rouge_l_scores)

    mean_chrf = np.mean(chrf_plus_plus_scores)
    sd_chrf = np.std(chrf_plus_plus_scores)

    n_samples = len(eval_data)
    
    # Confidence Intervals (95% CI)
    ci_bleu = stats.t.interval(0.95, n_samples - 1, loc=mean_bleu, scale=stats.sem(bleu_scores)) if np.any(bleu_scores) else (0.0, 0.0)
    ci_bert = stats.t.interval(0.95, n_samples - 1, loc=mean_bert, scale=stats.sem(bert_scores)) if np.any(bert_scores) else (0.0, 0.0)
    ci_rouge_l = stats.t.interval(0.95, n_samples - 1, loc=mean_rouge_l, scale=stats.sem(rouge_l_scores)) if np.any(rouge_l_scores) else (0.0, 0.0)
    ci_chrf = stats.t.interval(0.95, n_samples - 1, loc=mean_chrf, scale=stats.sem(chrf_plus_plus_scores)) if np.any(chrf_plus_plus_scores) else (0.0, 0.0)

    ci_bleu = [0.0 if np.isnan(x) or x < 0 else x for x in ci_bleu]
    ci_bert = [0.0 if np.isnan(x) else x for x in ci_bert]
    ci_rouge_l = [0.0 if np.isnan(x) or x < 0 else x for x in ci_rouge_l]
    ci_chrf = [0.0 if np.isnan(x) or x < 0 else x for x in ci_chrf]

    print("\n" + "=" * 70)
    print("📈 SMART HIV CARE: BILINGUAL PERFORMANCE METRIC REPORT")
    print(f"Evaluation Date:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Queries:      {n_samples}")
    print(f"Average Latency:    {df['latency'].mean():.2f}s")
    print(f"BLEU (Word-Wise):  {mean_bleu:.4f} ± {sd_bleu:.4f} (95% CI: {ci_bleu[0]:.4f}–{ci_bleu[1]:.4f})")
    print(f"ROUGE-L F1:        {mean_rouge_l:.4f} ± {sd_rouge_l:.4f} (95% CI: {ci_rouge_l[0]:.4f}–{ci_rouge_l[1]:.4f})")
    print(f"ChrF++ Score:      {mean_chrf:.4f} ± {sd_chrf:.4f} (95% CI: {ci_chrf[0]:.4f}–{ci_chrf[1]:.4f})")
    print(f"BERTScore F1:      {mean_bert:.4f} ± {sd_bert:.4f} (95% CI: {ci_bert[0]:.4f}–{ci_bert[1]:.4f})")
    print("=" * 70)

    output_file = "smarthivcare_eval_final.csv"
    df["bleu_score"] = bleu_scores
    df["rouge_l_score"] = rouge_l_scores
    df["chrf_plus_plus_score"] = chrf_plus_plus_scores
    df["bert_f1_score"] = bert_scores
    
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"💾 Comprehensive metrics log successfully saved to '{output_file}'\n")

if __name__ == "__main__":
    run_evaluation()
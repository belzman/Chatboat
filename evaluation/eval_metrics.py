import torch
import torch.nn as nn
import numpy as np
from evaluate import load
from sklearn.metrics import precision_recall_fscore_support

class BiLSTMClassifier(nn.Module):
    """
    Baseline BiLSTM architecture used for comparison in the manuscript.
    This represents the state-of-the-art for Amharic intent classification
    prior to RAG-LLM frameworks.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)

class ResearchEvaluator:
    def __init__(self):
        """Initializes SOTA evaluation metrics for low-resource NLP."""
        try:
            # BLEU: Measures syntactic precision
            self.bleu = load("bleu")
            # BERTScore: Measures semantic similarity using contextual embeddings
            self.bertscore = load("bertscore")
            print("✅ Evaluation Engines Loaded: BERTScore (Multilingual) & BLEU")
        except Exception as e:
            print(f"❌ Error loading metrics: {e}")

    def compute_comparative_metrics(self, y_true, y_pred):
        """Calculates Precision, Recall, and F1 for the BiLSTM baseline."""
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return {"Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4)}

    def compute_linguistic_quality(self, predictions, references):
        """
        Performs high-fidelity semantic evaluation.
        Utilizes bert-base-multilingual-cased to ensure Amharic morphological 
        nuances are captured during similarity calculation.
        """
        # Ensure input is not empty to prevent division by zero
        if not predictions or not predictions[0].strip():
            return {"BLEU": 0.0, "BERTScore_F1": 0.0}

        try:
            # 1. Syntactic Overlap (BLEU)
            bleu_res = self.bleu.compute(predictions=predictions, references=references)
            
            # 2. Semantic Embedding Similarity (BERTScore)
            # lang="am" is specified to optimize for Amharic tokenization
            bert_res = self.bertscore.compute(
                predictions=predictions, 
                references=references, 
                lang="am", 
                model_type="bert-base-multilingual-cased"
            )
            
            return {
                "BLEU": round(bleu_res['bleu'], 4),
                "BERTScore_F1": round(np.mean(bert_res['f1']), 4)
            }
        except Exception as e:
            print(f"⚠️ Metric Calculation Warning: {e}")
            return {"BLEU": 0.0, "BERTScore_F1": 0.0}
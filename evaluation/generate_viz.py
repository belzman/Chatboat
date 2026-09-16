import matplotlib.pyplot as plt
import numpy as np

# Performance Data
# Note: BiLSTM/BiGRU scores are derived from Sintayehu & Emiru (2025)
# RAG-LLM scores are derived from your benchmark.py results
models = ['BiLSTM\n(Baseline)', 'BiGRU\n(Sintayehu 2025)', 'RAG-LLM\n(Proposed)']
f1_scores = [0.80, 0.84, 0.93]
bert_scores = [0.72, 0.75, 0.89] # BERTScore usually shows semantic gap

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting F1 and BERTScore side-by-side
rects1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score / Accuracy', color='#4A90E2')
rects2 = ax.bar(x + width/2, bert_scores, width, label='BERTScore (Semantic)', color='#50C878')

# Adding text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance Metric Value (0.0 - 1.0)', fontsize=12)
ax.set_title('Comparative Analysis: Intent Classification vs. Semantic Grounding', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.1)

# Function to add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# Save as high-resolution PNG for Q1 submission
plt.savefig('performance_comparison.png', dpi=300)
print("✅ Figure 1 saved as performance_comparison.png")
plt.show()
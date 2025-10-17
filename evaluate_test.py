# ----------------------------------------------------------
# ✅ Evaluate Fine-tuned WangchanBERTa on Test Set
# ✅ Calculates Precision, Recall, F1-score and saves results
# ✅ Added elapsed time, batch encoding, and append to results file
# ----------------------------------------------------------

import os
import argparse
import pandas as pd
import torch
import time
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to wangchanberta-finetuned_lr2e-05 ")
parser.add_argument("--test_path", type=str, default="D:/thaisum_keyword/thaisum_test.csv", help="Path to test CSV")
parser.add_argument("--result_path", type=str, default="D:/thaisum_keyword/evaluation_results.csv", help="Path to save results")
parser.add_argument("--threshold", type=float, default=0.7, help="Cosine similarity threshold")
parser.add_argument("--sample_size", type=int, default=500, help="Number of test samples to evaluate (for speed)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding")
args = parser.parse_args()

# -----------------------------
# 1) Load model & test data
# -----------------------------
print("🚀 Loading fine-tuned model...")
model = SentenceTransformer(args.model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("📂 Loading test dataset...")
df = pd.read_csv(args.test_path)
df = df.dropna(subset=["body", "tags"])
df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
print(f"✅ Loaded {len(df)} test samples")

texts = df["body"].tolist()
true_keywords = df["tags"].tolist()

# -----------------------------
# 2) Evaluate
# -----------------------------
y_true, y_pred = [], []
threshold = args.threshold

print("⚙️ Evaluating...")
start_time = time.time()

# Batch encode texts and true keywords
text_embeddings = model.encode(texts, convert_to_tensor=True, batch_size=args.batch_size)
kw_embeddings = model.encode(true_keywords, convert_to_tensor=True, batch_size=args.batch_size)

for i in range(len(df)):
    text_emb = text_embeddings[i]
    kw_emb = kw_embeddings[i]

    # สุ่ม negative keyword
    neg_idx = i
    while neg_idx == i:
        neg_idx = torch.randint(0, len(df), (1,)).item()
    neg_kw_emb = kw_embeddings[neg_idx]

    # similarity
    pos_sim = util.cos_sim(text_emb, kw_emb).item()

    # ถ้า similarity ของจริงสูงกว่า threshold => ทายถูก
    pred_label = 1 if pos_sim > threshold else 0
    true_label = 1

    y_true.append(true_label)
    y_pred.append(pred_label)

end_time = time.time()
elapsed = end_time - start_time

# -----------------------------
# 3) Calculate metrics
# -----------------------------
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# -----------------------------
# 4) Save and append results
# -----------------------------
results = pd.DataFrame([{
    "Model": os.path.basename(args.model_path),
    "Threshold": threshold,
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1-Score": round(f1, 4),
    "Samples": len(df),
    "Elapsed_seconds": round(elapsed, 2),
    "Elapsed_minutes": round(elapsed/60, 2)
}])

# Append to existing results file or create new
if os.path.exists(args.result_path):
    df_existing = pd.read_csv(args.result_path)
    results = pd.concat([df_existing, results], ignore_index=True)

results.to_csv(args.result_path, index=False)

print("\n✅ Evaluation Complete!")
print(results)
print(f"\n📄 Results saved/appended to: {args.result_path}")
print(f"⏱ Total evaluation time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

# ----------------------------------------------------
# ✅ Fine-tune WangchanBERTa for Keyword Extraction (Safe + Fast Version)
# ✅ 3 Hyperparameters experiments
# ✅ Auto-handle embedding dimensions in evaluation
# ✅ Evaluate on test set with best hyperparameters
# ✅ Save each fine-tuned model separately
# ----------------------------------------------------

import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sklearn.metrics import accuracy_score, f1_score

# --------------------------
# Paths
# --------------------------
BASE_DIR = "D:/thaisum_keyword"
DATA_PATH = f"{BASE_DIR}/fine_tune_pairs.csv"
TEST_PATH = f"{BASE_DIR}/thaisum_test.csv"
MAIN_SAVE_DIR = f"{BASE_DIR}/wangchanberta_experiments"
os.makedirs(MAIN_SAVE_DIR, exist_ok=True)

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

text_col_candidates = ["body", "text", "title"]
keyword_col_candidates = ["tags", "keyword", "keywords"]

train_text_col = next((c for c in text_col_candidates if c in df.columns), None)
train_keyword_col = next((c for c in keyword_col_candidates if c in df.columns), None)

if train_text_col is None or train_keyword_col is None:
    raise ValueError("❌ CSV training file ต้องมีคอลัมน์ text/body/title และ tags/keyword")

df = df.dropna(subset=[train_text_col, train_keyword_col])
df = df[(df[train_text_col].str.strip() != "") & (df[train_keyword_col].str.strip() != "")]

# --------------------------
# Use subset for faster training
# --------------------------
subset_size = 50000  # ใช้ 50k rows
if len(df) > subset_size:
    df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)

texts = df[train_text_col].astype(str).tolist()
keywords = df[train_keyword_col].astype(str).tolist()
train_examples = [InputExample(texts=[t, k], label=1.0) for t, k in zip(texts, keywords)]
print(f"✅ Training subset ready: {len(train_examples)} rows")

# --------------------------
# Load test dataset
# --------------------------
if os.path.exists(TEST_PATH):
    df_test = pd.read_csv(TEST_PATH)
    df_test.columns = [c.strip().lower() for c in df_test.columns]

    test_text_col = next((c for c in text_col_candidates if c in df_test.columns), None)
    test_keyword_col = next((c for c in keyword_col_candidates if c in df_test.columns), None)

    df_test = df_test.dropna(subset=[test_text_col, test_keyword_col])
    test_texts = df_test[test_text_col].astype(str).tolist()
    test_keywords = df_test[test_keyword_col].astype(str).tolist()
    print(f"✅ Test dataset ready: {len(test_texts)} rows")
else:
    df_test = None
    print("⚠️ Test file not found, skipping evaluation.")

# --------------------------
# Hyperparameters experiments
# --------------------------
experiments = [
    {"lr": 2e-5, "batch_size": 16, "max_seq_len": 128},
    {"lr": 3e-5, "batch_size": 32, "max_seq_len": 128},
    {"lr": 5e-5, "batch_size": 16, "max_seq_len": 256},
]

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {device}")
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Running on CPU (จะช้ากว่า GPU)")

# --------------------------
# Validation split
# --------------------------
val_ratio = 0.1
val_size = int(len(train_examples) * val_ratio)
train_size = len(train_examples) - val_size
train_dataset, val_dataset = random_split(train_examples, [train_size, val_size])

# --------------------------
# Safe Evaluation Function
# --------------------------
def evaluate_model(model, texts, keywords, device, threshold=0.5):
    model.eval()
    sims, y_true, y_pred = [], [], []

    with torch.no_grad():
        for t, k in zip(texts, keywords):
            t_emb = model.encode(t, convert_to_tensor=True, device=device)
            k_emb = model.encode(k, convert_to_tensor=True, device=device)

            # ✅ ป้องกันมิติไม่ตรง
            if t_emb.dim() == 1:
                t_emb = t_emb.unsqueeze(0)
            if k_emb.dim() == 1:
                k_emb = k_emb.unsqueeze(0)

            sim = torch.nn.functional.cosine_similarity(t_emb, k_emb).item()
            sims.append(sim)
            y_true.append(1)
            y_pred.append(1 if sim >= threshold else 0)

    avg_sim = sum(sims) / len(sims)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return avg_sim, acc, f1

# --------------------------
# Training & validation
# --------------------------
results, val_results = [], []
best_exp = None
best_f1 = -1.0

for exp_id, params in enumerate(experiments, start=1):
    print(f"\n==============================")
    print(f"🧪 Experiment {exp_id}: {params}")
    print(f"==============================")

    # Model
    transformer_model = models.Transformer(
        "airesearch/wangchanberta-base-att-spm-uncased",
        max_seq_length=params["max_seq_len"]
    )
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[transformer_model, pooling_model])
    model.to(device)

    # DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=params["batch_size"])
    val_dataloader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # Loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Train
    print("⏳ Training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=0,
        optimizer_params={'lr': params["lr"]},
        show_progress_bar=True
    )
    print("✅ Training finished")

    # ✅ Save model for this experiment
    model_save_path = f"{BASE_DIR}/wangchanberta-finetuned_lr{str(params['lr']).replace('.', '')}"
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}")

    # Validation
    print("🔍 Evaluating validation set...")
    avg_sim, acc, f1 = evaluate_model(
        model,
        [ex.texts[0] for ex in val_dataset],
        [ex.texts[1] for ex in val_dataset],
        device
    )
    print(f"📝 Validation - Exp {exp_id}: Avg Sim={avg_sim:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    val_results.append({
        "Experiment": exp_id,
        "Learning Rate": params["lr"],
        "Batch Size": params["batch_size"],
        "Max Seq Length": params["max_seq_len"],
        "Val Avg Cosine Sim": round(avg_sim, 4),
        "Val Accuracy": round(acc, 4),
        "Val F1": round(f1, 4),
        "Saved Model Path": model_save_path
    })

    if f1 > best_f1:
        best_f1 = f1
        best_exp = (exp_id, params, model, model_save_path)

# Save validation summary
pd.DataFrame(val_results).to_csv(f"{MAIN_SAVE_DIR}/validation_summary.csv", index=False)

# --------------------------
# Evaluate test set with best hyperparameters
# --------------------------
if df_test is not None and best_exp:
    exp_id, params, best_model, best_model_path = best_exp
    print(f"\n🏁 Testing with best model (Exp {exp_id}) ...")
    avg_sim, acc, f1 = evaluate_model(best_model, test_texts, test_keywords, device)
    print(f"✅ Test Results (Best Exp {exp_id}): Avg Sim={avg_sim:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    pd.DataFrame([{
        "Experiment": exp_id,
        "Learning Rate": params["lr"],
        "Batch Size": params["batch_size"],
        "Max Seq Length": params["max_seq_len"],
        "Test Avg Cosine Sim": round(avg_sim, 4),
        "Test Accuracy": round(acc, 4),
        "Test F1": round(f1, 4),
        "Best Model Path": best_model_path
    }]).to_csv(f"{MAIN_SAVE_DIR}/test_summary.csv", index=False)

print("\n📊 Training, validation, and test evaluation completed without errors ✅")

# --------------------------------------- 
# Thai Keyword Extraction + F1 Evaluation 
# Using TF-IDF, KeyBERT, WangchanBERTa 
# Ground truth: tags 
# --------------------------------------- 
 
import os 
import pandas as pd 
from tqdm import tqdm 
from pythainlp.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer 
from keybert import KeyBERT 
from transformers import AutoModel, AutoTokenizer 
 
# ------------------------------ 
# 0) ตั้งค่า Cache / Temp Folder 
# ------------------------------ 
BASE_DIR = "D:/thaisum_keyword" 
os.environ['TRANSFORMERS_CACHE'] = f"{BASE_DIR}/huggingface_cache" 
os.environ['TMPDIR'] = f"{BASE_DIR}/temp" 
os.makedirs(f"{BASE_DIR}/huggingface_cache", exist_ok=True) 
os.makedirs(f"{BASE_DIR}/temp", exist_ok=True) 
 
# ------------------------------ 
# 1) โหลด Dataset 
# ------------------------------ 
df = pd.read_csv(f"{BASE_DIR}/thaisum_test.csv") 
print(f"✅ Loaded {len(df)} rows from CSV.\n") 
print(df.head(3)) 
 
# ------------------------------ 
# 2) Preprocess (ตัดคำไทย) 
# ------------------------------ 
def preprocess(text): 
    tokens = word_tokenize(str(text), keep_whitespace=False) 
    return " ".join(tokens) 
 
print("\n🔹 Preprocessing text (tokenizing Thai)...") 
df["body_clean"] = df["body"].astype(str).apply(preprocess) 
 
# ------------------------------ 
# 3) TF-IDF Keyword Extraction 
# ------------------------------ 
def extract_tfidf_keywords(texts, top_n=5): 
    vectorizer = TfidfVectorizer(max_features=5000) 
    X = vectorizer.fit_transform(texts) 
    feature_names = vectorizer.get_feature_names_out() 
    results = [] 
    for i, row in enumerate(tqdm(X, desc="TF-IDF extraction")): 
        doc = row.toarray().flatten() 
        indices = doc.argsort()[-top_n:][::-1] 
        keywords = [feature_names[i] for i in indices] 
        results.append(keywords) 
    return results 
 
tfidf_keywords = extract_tfidf_keywords(df["body_clean"], top_n=5) 
 
# ------------------------------ 
# 4) KeyBERT (Multilingual model) 
# ------------------------------ 
print("\n🔹 Extracting with KeyBERT (multilingual)...") 
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2') 
 
keybert_keywords = [] 
for text in tqdm(df["body_clean"], desc="KeyBERT extraction"): 
    keywords = kw_model.extract_keywords( 
        text, 
        keyphrase_ngram_range=(1, 1), 
        stop_words=None, 
        top_n=5 
    ) 
    keybert_keywords.append([k[0] for k in keywords]) 
 
# ------------------------------ 
# 5) KeyBERT + WangchanBERTa 
# ------------------------------ 
print("\n🔹 Extracting with WangchanBERTa (Thai model)...") 
tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased") 
model = AutoModel.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased") 
kw_model_thai = KeyBERT(model=model) 
 
wangchan_keywords = [] 
for text in tqdm(df["body_clean"], desc="WangchanBERTa extraction"): 
    keywords = kw_model_thai.extract_keywords( 
        text, 
        keyphrase_ngram_range=(1, 1), 
        stop_words=None, 
        top_n=5 
    ) 
    wangchan_keywords.append([k[0] for k in keywords]) 
 
# ------------------------------ 
# 6) เตรียม Ground Truth (tags) 
# ------------------------------ 
print("\n🔹 Preparing ground truth keywords (tags)...") 
df["true_keywords"] = df["tags"].fillna("").apply( 
    lambda x: [t.strip() for t in str(x).split(",") if t.strip() != ""] 
) 
 
# ------------------------------ 
# 7) สร้างตารางเปรียบเทียบ 
# ------------------------------ 
comparison = pd.DataFrame({ 
    "text": df["body"], 
    "tfidf": tfidf_keywords, 
    "keybert": keybert_keywords, 
    "wangchanberta": wangchan_keywords, 
    "true_keywords": df["tags"] 
}) 
 
comparison["true_keywords"] = comparison["true_keywords"].fillna("") 
 
# แปลง tags เป็น list เช่น "การเมือง, รัฐธรรมนูญ" -> ["การเมือง", "รัฐธรรมนูญ"] 
comparison["true_keywords"] = comparison["true_keywords"].apply( 
    lambda x: [w.strip() for w in str(x).split(",") if w.strip()] 
) 
 
# ------------------------------ 
# 8) คำนวณ F1-score 
# ------------------------------ 
def f1_score_lists(pred, true): 
    pred_set = set(pred) 
    true_set = set(true) 
    if len(true_set) == 0: 
        return 0.0 
    tp = len(pred_set & true_set) 
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0 
    recall = tp / len(true_set) if len(true_set) > 0 else 0 
    if precision + recall == 0: 
        return 0.0 
    return 2 * (precision * recall) / (precision + recall) 
 
print("\n🔹 Calculating F1-scores...") 
f1_tfidf = comparison.apply(lambda row: f1_score_lists(row["tfidf"], row["true_keywords"]), axis=1).mean() 
f1_keybert = comparison.apply(lambda row: f1_score_lists(row["keybert"], row["true_keywords"]), axis=1).mean() 
f1_wangchan = comparison.apply(lambda row: f1_score_lists(row["wangchanberta"], row["true_keywords"]), axis=1).mean() 
 
print("\n📊 Average F1-scores:") 
print(f"TF-IDF: {f1_tfidf:.4f}") 
print(f"KeyBERT: {f1_keybert:.4f}") 
print(f"WangchanBERTa: {f1_wangchan:.4f}") 
 
# ------------------------------ 
# 9) Export results 
# ------------------------------ 
summary = pd.DataFrame({ 
    "Method": ["TF-IDF", "KeyBERT", "WangchanBERTa"], 
    "F1-score": [f1_tfidf, f1_keybert, f1_wangchan] 
}) 
  
comparison.to_csv(f"{BASE_DIR}/keywords_all_methods.csv", index=False, encoding="utf-8-sig") 
comparison.to_excel(f"{BASE_DIR}/keywords_all_methods.xlsx", index=False) 
summary.to_csv(f"{BASE_DIR}/f1_scores.csv", index=False, encoding="utf-8-sig") 
 
print("\n✅ Export finished") 
print(summary)
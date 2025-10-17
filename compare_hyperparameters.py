# -------------------------------------------------------
# ✅ Compare Results from Multiple Fine-tuned Models
# ✅ ใช้ไฟล์ evaluation_results.csv ที่รวมผลทุกโมเดล
# -------------------------------------------------------

import os
import pandas as pd

# 🔧 Path ของไฟล์ผลลัพธ์รวมทุกโมเดล
RESULT_FILE = "D:/thaisum_keyword/evaluation_results.csv"

# -----------------------------
# 1) ตรวจสอบว่าไฟล์มีอยู่
# -----------------------------
if not os.path.exists(RESULT_FILE):
    print(f"❌ ไม่พบไฟล์ {RESULT_FILE}")
    exit()

# -----------------------------
# 2) โหลดผลลัพธ์
# -----------------------------
df = pd.read_csv(RESULT_FILE)

if df.empty:
    print("❌ ไฟล์ผลลัพธ์ว่าง ไม่มีข้อมูลให้เปรียบเทียบ")
    exit()

# -----------------------------
# 3) แสดงผลรวมทั้งหมด
# -----------------------------
print("\n📊 All Models Results:")
print(df)

# -----------------------------
# 4) หาค่าโมเดลที่ดีที่สุดตาม F1-Score
# -----------------------------
best_model = df.loc[df["F1-Score"].idxmax()]

print("\n✅ Best Model (by F1-Score):")
print(best_model)

# -----------------------------
# 5) บันทึกผลรวมทั้งหมดเป็น CSV ใหม่ (optional)
# -----------------------------
save_path = os.path.join(os.path.dirname(RESULT_FILE), "hyperparameter_comparison.csv")
df.to_csv(save_path, index=False)

print(f"\n📁 Saved comparison table to: {save_path}")

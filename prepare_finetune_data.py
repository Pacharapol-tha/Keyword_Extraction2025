import pandas as pd

# -----------------------
# ตั้งค่า path ของไฟล์
# -----------------------
BASE_DIR = "D:/thaisum_keyword"   # โฟลเดอร์หลักของคุณ
INPUT_FILE = f"{BASE_DIR}/thaisum_train.csv"
OUTPUT_FILE = f"{BASE_DIR}/fine_tune_pairs.csv"

# -----------------------
# โหลดข้อมูล
# -----------------------
df = pd.read_csv(INPUT_FILE)
print(f"✅ โหลดข้อมูล {len(df)} แถวจากไฟล์ {INPUT_FILE}")

# -----------------------
# ตรวจสอบว่ามีคอลัมน์ body / tags ไหม
# -----------------------
print("\nคอลัมน์ทั้งหมดใน dataset:")
print(df.columns.tolist())

# -----------------------
# สร้างคู่ (text, keyword)
# -----------------------
pairs = []

for _, row in df.iterrows():
    body = str(row.get("body", "")).strip()
    tags = str(row.get("tags", "")).strip()
    if not body or not tags:
        continue
    for tag in tags.split(","):
        tag_clean = tag.strip()
        if tag_clean:
            pairs.append({"text": body, "keyword": tag_clean})

df_pairs = pd.DataFrame(pairs)
print(f"\n✅ ได้คู่ text-keyword จำนวน: {len(df_pairs):,} ตัวอย่าง")

# -----------------------
# ✅ บันทึกไฟล์ใหม่
# -----------------------
df_pairs.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ บันทึกไฟล์ใหม่เรียบร้อย: {OUTPUT_FILE}")

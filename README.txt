🧠 README.txt — Fine-tune WangchanBERTa for Keyword Extraction

🔍 Overview
โครงงานนี้เป็นการ Fine-tune โมเดล WangchanBERTa สำหรับงาน Keyword Extraction (ดึงคำสำคัญจากบทความภาษาไทย)
โดยใช้ข้อมูลจาก ThaiSum และรันได้ครบตั้งแต่เตรียมข้อมูล → เทรน → ประเมินผล → เปรียบเทียบ hyperparameters

------------------------------------------------------------
⚙️ Requirements

1. ติดตั้ง Python และ Library ที่ใช้
Python เวอร์ชันแนะนำ: 3.9 - 3.11

ติดตั้ง dependencies ด้วยคำสั่ง:
pip install pandas torch sentence-transformers scikit-learn
/*pip install dependencies ส่วนที่เหลือ*/

หากต้องการใช้ GPU ให้ติดตั้ง PyTorch เวอร์ชันที่รองรับ CUDA
ดูได้ที่ https://pytorch.org/get-started/locally/

------------------------------------------------------------
📂 Folder Structure

D:/thaisum_keyword/
│
├── thaisum_train.csv          
├── thaisum_test.csv  
├── thaisum_validation.csv          
│
├── keyword_extraction.py
├── prepare_tune_data.py       
├── fine_tune_wangchanberta.py 
├── evaluate_test.py           
├── compare_hyperparameters.py 
│
├── fine_tune_pairs.csv        
├── evaluation_results.csv     
└── wangchanberta_experiments/ 

------------------------------------------------------------
🧩 ขั้นตอนการรันทั้งหมด

▶️ ขั้นตอนที่ 1: ลองรัน keyword_extraction.py เพื่อให้ได้ไฟล์ที่แสดง output และ input กับค่า f1-score ก่อนที่จะทำการ fine tune WangchanBERTa 
จะได้ไฟล์ keywords_all_methods.csv กับ keywords_all_methods.xlsx และ f1_scores.csv

▶️ ขั้นตอนที่ 2: เตรียมข้อมูลสำหรับ Fine-tune
python prepare_tune_data.py จะได้ไฟล์ fine_tune_pairs.csv เพื่อนำไปใช้ fine tune WangchanBERTa

▶️ ขั้นตอนที่ 3: Fine-tune WangchanBERTa
python fine_tune_wangchanberta.py

จะเทรน 2 ชุด hyperparameters:
1. lr=2e-5, batch=16, max_len=128
2. lr=3e-5, batch=32, max_len=128

เซฟโมเดลใน:
wangchanberta-finetuned_lr2e05/
wangchanberta-finetuned_lr3e05/


▶️ ขั้นตอนที่ 3: ประเมินผลแต่ละโมเดล
python evaluate_test.py --model_path "D:/thaisum_keyword/wangchanberta-finetuned_lr2e05"
python evaluate_test.py --model_path "D:/thaisum_keyword/wangchanberta-finetuned_lr3e05"

Argument ที่ใช้:
--model_path       path ของโมเดลที่ fine-tune แล้ว
--test_path        path ของ test set
--result_path      path สำหรับเซฟผลลัพธ์
--threshold        ค่าความคล้าย (cosine similarity)
--sample_size      จำนวนตัวอย่างทดสอบ
--batch_size       ขนาด batch

▶️ ขั้นตอนที่ 4: เปรียบเทียบผลลัพธ์ทั้งหมด
python compare_hyperparameters.py

------------------------------------------------------------
🧾 สรุป Output สำคัญ

| ไฟล์ | หน้าที่ | ที่อยู่ |
|-------|-----------|-----------|
| fine_tune_pairs.csv | คู่ข้อมูล text-keyword | D:/thaisum_keyword/ |
| wangchanberta-finetuned_lr*. | โมเดลแต่ละชุด | D:/thaisum_keyword/ |
| validation_summary.csv | ผล validation | D:/thaisum_keyword/wangchanberta_experiments/ |
| test_summary.csv | ผล test set | D:/thaisum_keyword/wangchanberta_experiments/ |
| evaluation_results.csv | รวมผล evaluate | D:/thaisum_keyword/ |
| hyperparameter_comparison.csv | สรุปเปรียบเทียบ | D:/thaisum_keyword/ |

------------------------------------------------------------
💡 Tips
- หากใช้ GPU ตรวจสอบ torch.cuda.is_available() == True
- เพิ่ม epochs ได้ใน fine_tune_wangchanberta.py (epochs=1 → 2)
- หากข้อมูลใหญ่ ให้ลด subset_size

------------------------------------------------------------
📚 Reference
- WangchanBERTa: https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased
- Sentence Transformers: https://www.sbert.net
- ThaiSum Dataset: https://www.kaggle.com/datasets/ratthachat/nakhunchumpolsathien-thaisum

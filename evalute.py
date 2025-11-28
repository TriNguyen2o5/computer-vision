import subprocess
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Thông số ---
# LẤY ĐƯỜNG DẪN model "best.pt" TỪ KẾT QUẢ TRAIN TRƯỚC
MODEL_PATH = "runs/detect/train_tomato_pest_v8n/weights/best.pt"
DATASET_YAML = "pest_final/data.yaml"
IMGSZ = 640 # Phải dùng đúng imgsz đã train

# --- Lệnh test YOLO ---
cmd = [
    "yolo",
    "detect",
    "val",  
    f"model={MODEL_PATH}",
    f"data={DATASET_YAML}",
    f"imgsz={IMGSZ}",
    "split=test", # <-- THAM SỐ QUAN TRỌNG NHẤT
    "device=0", 
    "verbose=True",
    "save_json=True",
    "plots=True",
    "project=runs/evaluate",           # thư mục cha
    "name=test_pest_v8n" 
]

print("Bắt đầu TEST mô hình trên tập test...")
subprocess.run(cmd, check=True)
print("Hoàn tất test!")
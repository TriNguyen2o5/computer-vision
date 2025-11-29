import subprocess
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Thông số
MODEL_DET = "yolov8n.pt"
DATASET_PEST_YAML = "D:/ThiGiacMayTinh/CV-main/pest_final/data.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 8

# Lệnh train YOLO leaf detection
cmd = [
    "yolo",
    "detect",
    "train",
    f"model={MODEL_DET}",
    f"data={DATASET_PEST_YAML}",
    f"epochs={EPOCHS}",
    f"imgsz={IMGSZ}",
    f"batch={BATCH}",
    "device=0", 
    "verbose=True",
    "name=train_tomato_pest_v8n"
]

print("Bắt đầu train YOLO pest detection...")
subprocess.run(cmd, check=True)
print("Hoàn tất train YOLO pest detection!")

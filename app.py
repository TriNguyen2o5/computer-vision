import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import pillow_heif
from PIL import Image
import os
import tempfile

# Import hàm xử lý ảnh TỔNG QUÁT
from image_processor import preprocess_image_for_yolo

# Đăng ký trình mở file HEIF (cho ảnh iPhone)
pillow_heif.register_heif_opener()
# Tối ưu cho Streamlit
torch.set_num_threads(1)

# ==========================================
#  Cấu hình trang
# ==========================================
st.set_page_config(page_title="Leaf Disease Detection", layout="wide")
st.title("Phát hiện Bệnh & Sâu bọ (YOLOv8)")

# ==========================================
#  Tải Model (Chỉ tải 1 lần)
# ==========================================
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Đang tải models lên **{device.upper()}**...")
    
    try:
        model_disease = YOLO("runs/detect/train_tomato_leaf_v8m/weights/best.pt").to(device)
        model_pest = YOLO("runs/detect/train_tomato_pest_v8n/weights/best.pt").to(device)
        st.sidebar.success(f" Models đã sẵn sàng trên **{device.upper()}**!")
        return model_disease, model_pest, device
    except Exception as e:
        st.sidebar.error(f"Lỗi tải model: {e}")
        st.stop()

disease_detect, pest_dect, device = load_models()

# ==========================================
# HÀM XỬ LÝ ẢNH CHUNG
# ==========================================
def process_and_draw_boxes(input_image_bgr, enable_disease=True, enable_pest=True, conf_thresh=0.25, iou_thresh=0.45):
    """
    Hàm này nhận ảnh BGR, chạy các model YOLO,
    và vẽ các bounding box trực tiếp lên ảnh đó.
    """
    
    image_with_boxes = input_image_bgr.copy()
    summary = []
    
    # --- 1. Chạy Model Bệnh lá ---
    if enable_disease:
        disease_results = disease_detect(input_image_bgr, verbose=False, device=device,conf=conf_thresh, iou=iou_thresh)
        disease_names = disease_detect.names
        color = (0, 255, 0) # Xanh lá
        
        for r in disease_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                label = f"{disease_names[cls]} {conf*100:.1f}%"
                summary.append(f"[Bệnh] {label}")
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- 2. Chạy Model Sâu bọ ---
    if enable_pest:
        pest_results = pest_dect(input_image_bgr, verbose=False, device=device,conf=conf_thresh, iou=iou_thresh)
        pest_names = pest_dect.names
        color = (0, 100, 255) # Cam
        
        for r in pest_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                label = f"{pest_names[cls]} {conf*100:.1f}%"
                summary.append(f"[Sâu] {label}")
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
    return image_with_boxes, summary

# ==========================================
#  Giao diện Sidebar
# ==========================================
st.sidebar.header("Cài đặt Model")
enable_dect = st.sidebar.checkbox("Phát hiện Bệnh lá", value=True, key="cb_disease")
enable_pest = st.sidebar.checkbox("Phát hiện Sâu bọ", value=True, key="cb_pest")

st.sidebar.header("Cài đặt Xử lý Ảnh")
conf_threshold = st.sidebar.slider("Ngưỡng phát hiện (Confidence)", 
                                    min_value=0.0, max_value=1.0, 
                                    value=0.25, step=0.05,
                                    key="conf_slider",
                                    help="Lọc bỏ các phát hiện có độ tin cậy thấp.")

# *** THÊM MỚI ***: Thêm thanh trượt IOU
iou_threshold = st.sidebar.slider("Ngưỡng chồng lấn (IOU)",
                                   min_value=0.0, max_value=1.0,
                                   value=0.45, step=0.05,
                                   key="iou_slider",
                                   help="Lọc bỏ các ô vuông bị trùng lặp. Giá trị thấp = lọc nghiêm ngặt hơn.")

# ==========================================
# 🖥 Giao diện 2 nút
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.header("1. Tải ảnh lên")
    uploaded_file = st.file_uploader("Tải file từ máy", type=["jpg","jpeg","png","bmp","webp","tiff","jfif","heic"], key="uploader")

with col2:
    st.header("2. Chụp ảnh (Webcam)")
    camera_file = st.camera_input("Chụp ảnh bằng camera", key="camera")

# ==========================================
# 🖼 Xử lý và Dự đoán
# ==========================================

# Xác định nguồn ảnh
source_file = uploaded_file or camera_file
image_bgr_processed = None
image_pil_display = None

if source_file is not None:
    # 1. Lưu file tạm
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(source_file.read())
    image_path = tfile.name
    
    st.write("---")
    
    # 2. GỌI HÀM XỬ LÝ CHUNG
    # Truyền giá trị của checkbox `apply_fix` vào
    image_bgr_processed, image_pil_display = preprocess_image_for_yolo(
        image_path)
    
    # 3. Dọn dẹp file tạm
    tfile.close()
    os.remove(image_path)
    
    # --- Chỉ chạy dự đoán NẾU CÓ ảnh đã xử lý ---
    if image_bgr_processed is not None and image_pil_display is not None:
        
        with st.spinner('Model đang phát hiện...'):
            start_time = time.time()
            # 4. Gọi hàm dự đoán chung
            result_img, summary = process_and_draw_boxes(
                image_bgr_processed, 
                enable_dect, 
                enable_pest,
                conf_thresh=conf_threshold, # Lấy giá trị từ thanh trượt
                iou_thresh=iou_threshold   # Lấy giá trị từ thanh trượt
            )
            end_time = time.time()

        # --- 5. Hiển thị ---
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.image(image_pil_display, 
                     caption="Ảnh gốc (để so sánh)", 
                     use_column_width=True)
        
        with col_res2:
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                     caption="Kết quả phát hiện", 
                     use_column_width=True)
        
        if summary:
            st.success("**Kết quả:** " + " | ".join(summary))
        else:
            if not enable_dect and not enable_pest:
                st.warning("Bạn đã tắt cả hai model. Vui lòng bật ít nhất một model trong Sidebar.")
            else:
                st.info("Không phát hiện đối tượng nào (với ngưỡng Conf > " f"{conf_threshold*100:.1f}%" " và IOU < " f"{iou_threshold*100:.1f}%).")
        
        st.caption(f"Thời gian xử lý: {end_time - start_time:.2f} giây trên {device.upper()}")
            
    else:
        st.error(" Không thể đọc hoặc xử lý file ảnh. Vui lòng thử ảnh khác.")
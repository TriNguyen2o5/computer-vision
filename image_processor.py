import numpy as np
import cv2
from PIL import Image


def preprocess_image_for_yolo(image_path_or_frame, apply_enhancements=False):
    """
    Hàm xử lý TỔNG QUÁT (cho cả Upload và Camera).
    1. Đọc ảnh (an toàn Unicode).
    2. Trả về:
       - img_bgr_processed (Numpy array BGR): Ảnh đã xử lý (hoặc ảnh gốc) để đưa vào YOLO.
       - img_pil_display (PIL.Image RGB): Ảnh gốc (chưa xử lý) để hiển thị so sánh.
    """
    frame = None
    original_pil_image = None
    
    try:
        if isinstance(image_path_or_frame, str):
            # 1. Đọc bằng PIL (an toàn với Unicode/HEIC)
            original_pil_image = Image.open(image_path_or_frame).convert('RGB')
            frame = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
        else:
            frame = image_path_or_frame
            original_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return frame, original_pil_image

    except Exception as e:
        print(f"Lỗi trong image_processor (YOLO): {e}")
        return None, None
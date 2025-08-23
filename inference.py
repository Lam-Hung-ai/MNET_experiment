import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from argparse import Namespace, ArgumentParser
import os 
from model.MNetold import MNetold

# ... (Hàm preprocess_image, postprocess_and_save, run_inference_refactored giữ nguyên như cũ) ...

def preprocess_image(image_path, device):
    """
    Xử lý ảnh đầu vào: đọc, chuyển màu, resize và chuyển thành tensor.

    :param image_path: Đường dẫn đến ảnh đầu vào.
    :param device: Thiết bị (CPU hoặc GPU) để chuyển tensor đến.
    :return: Tensor của ảnh đã được xử lý, sẵn sàng cho model.
    """
    print(f">> Đang xử lý ảnh đầu vào: {image_path}")
    
    # Đọc ảnh bằng OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không thể tìm thấy hoặc đọc ảnh tại: {image_path}")
    
    # Chuyển đổi từ BGR (mặc định của OpenCV) sang RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize ảnh về 256x256
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    # Chuyển ảnh NumPy (H, W, C) thành Tensor (C, H, W) và chuẩn hóa về [0, 1]
    tensor = transforms.ToTensor()(img_resized)
    
    # Thêm chiều batch (B, C, H, W) và chuyển đến thiết bị
    return tensor.unsqueeze(0).to(device)

def postprocess_and_save(restored_bg, mask, original_input_tensor, output_path):
    """
    Hậu xử lý kết quả từ model và lưu ảnh cuối cùng.

    :param restored_bg: Tensor nền đã khôi phục từ model.
    :param mask: Tensor mặt nạ từ model.
    :param original_input_tensor: Tensor gốc đầu vào (dùng để ghép ảnh).
    :param output_path: Đường dẫn để lưu ảnh kết quả.
    """
    print(">> Đang hậu xử lý kết quả...")

    # Áp dụng công thức để tạo ảnh cuối cùng
    final_output_tensor = restored_bg * mask + original_input_tensor * (1.0 - mask)
    
    # Lấy tensor từ batch và chuyển sang CPU
    output_tensor = final_output_tensor.squeeze(0).cpu()
    
    # Chuyển từ (C, H, W) sang (H, W, C)
    output_numpy_rgb = output_tensor.permute(1, 2, 0).numpy()
    
    # Chuyển giá trị pixel từ [0, 1] về [0, 255] và định dạng uint8
    output_numpy_rgb = (output_numpy_rgb * 255).astype(np.uint8)

    # Chuyển đổi lại từ RGB sang BGR để OpenCV lưu
    output_numpy_bgr = cv2.cvtColor(output_numpy_rgb, cv2.COLOR_RGB2BGR)

    # Lưu ảnh bằng OpenCV
    cv2.imwrite(output_path, output_numpy_bgr)
    print(f">> Đã lưu ảnh kết quả tại: {output_path}")

def run_inference_refactored(model_path, image_path, output_path):
    """
    Hàm chính để điều phối quá trình inference.
    """
    # --- Cấu hình và nạp model ---
    args = Namespace(k1=5, k2=2, k3=1, srm=1, inf=1, crf=1, sharing=1)
    print(">> Khởi tạo model với kiến trúc MNet...")
    model = MNetold(args=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Sử dụng thiết bị: {device}")

    print(f">> Nạp trọng số từ: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # --- Xử lý, Inference và Lưu ---
    
    # 1. Xử lý ảnh đầu vào
    input_tensor = preprocess_image(image_path, device)
    
    # 2. Thực hiện Inference
    print(">> Bắt đầu quá trình inference...")
    with torch.no_grad():
        restored_bg, mask = model(input_tensor)

    # 3. Hậu xử lý và lưu kết quả
    postprocess_and_save(restored_bg, mask, input_tensor, output_path)

# --- Cấu hình và chạy ---
if __name__ == '__main__':
    # --- Tạo trình phân tích đối số từ dòng lệnh ---
    parser = ArgumentParser(description='Chạy inference cho model MNetold.')
    
    # --- Định nghĩa các đối số cần nhận ---
    parser.add_argument('-i', '--image', type=str, required=True, help='Đường dẫn đến ảnh đầu vào.')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Đường dẫn đến file trọng số model (.pth.tar).')
    
    # --- Đọc các đối số từ dòng lệnh ---
    args = parser.parse_args()

    # --- Cấu hình và chạy ---
    MODEL_CHECKPOINT = args.weights
    INPUT_IMAGE = args.image

    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Lấy tên file từ đường dẫn input để đặt tên cho file output
    image_name = os.path.basename(INPUT_IMAGE)

    final_output_path = os.path.join(RESULTS_DIR, image_name)
    
    # Gọi hàm chính với các đường dẫn đã nhận được
    run_inference_refactored(MODEL_CHECKPOINT, INPUT_IMAGE, final_output_path)
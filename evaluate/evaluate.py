import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import piq
import argparse
from model.MNetold import MNetold
from tqdm import tqdm
import json
import numpy as np

def preprocess_image(image_path, device):
    """
    Hàm này nhận đường dẫn của một ảnh, sau đó đọc và chuyển đổi nó
    thành định dạng tensor mà model AI có thể hiểu được.

    Args:
        image_path (str): Đường dẫn tới file ảnh cần xử lý.
        device (torch.device): Thiết bị tính toán (CPU hoặc GPU) mà tensor sẽ được gửi đến.

    Returns:
        torch.Tensor: Một tensor biểu diễn ảnh đã được xử lý.
    """
    # Sử dụng khối try-except để bắt lỗi nếu không tìm thấy file
    try:
        # Mở file ảnh bằng thư viện Pillow
        img = Image.open(image_path)
    except FileNotFoundError:
        # Nếu không tìm thấy, báo lỗi và dừng chương trình
        raise FileNotFoundError(f"Lỗi: Không thể tìm thấy ảnh tại đường dẫn '{image_path}'")

    # Chuyển ảnh sang hệ màu RGB. Bước này đảm bảo mọi ảnh đầu vào
    # (dù là RGBA, ảnh xám,...) đều được đưa về một định dạng 3 kênh màu nhất quán.
    img_rgb = img.convert("RGB")

    # Thay đổi kích thước ảnh về 256x256 pixels. Đây là kích thước đầu vào
    # cố định mà model này yêu cầu.
    # Image.Resampling.LANCZOS là một thuật toán giảm/phóng kích thước cho chất lượng cao.
    img_resized = img_rgb.resize((256, 256), Image.Resampling.LANCZOS)

    # Chuyển ảnh PIL thành PyTorch Tensor.
    # transforms.ToTensor() tự động thực hiện 2 việc:
    # 1. Chuyển cấu trúc từ (H, W, C) sang (C, H, W) mà PyTorch yêu cầu.
    # 2. Chuẩn hóa giá trị của mỗi pixel từ [0, 255] về khoảng [0.0, 1.0].
    tensor = transforms.ToTensor()(img_resized)

    # Thêm một chiều "batch" vào đầu tensor (từ C,H,W -> B,C,H,W với B=1).
    # Model luôn xử lý dữ liệu theo lô (batch), nên dù chỉ có 1 ảnh, ta vẫn cần chiều này.
    # Sau đó, chuyển tensor lên thiết bị đã chọn (CPU hoặc GPU).
    return tensor.unsqueeze(0).to(device)


#  HÀM HẬU XỬ LÝ VÀ LƯU KẾT QUẢ

def postprocess_and_save(restored_bg, mask, original_input_tensor, output_path):
    """
    Hàm này nhận kết quả đầu ra của model, kết hợp chúng để tạo thành ảnh cuối cùng
    và lưu ảnh đó xuống đĩa.

    Args:
        restored_bg (torch.Tensor): Tensor chứa ảnh nền đã được model tái tạo.
        mask (torch.Tensor): Tensor mặt nạ (mask) do model tạo ra.
        original_input_tensor (torch.Tensor): Tensor của ảnh gốc ban đầu.
        output_path (str): Đường dẫn để lưu file ảnh kết quả.
    """

    # Công thức ghép ảnh:
    # Lấy phần nền đã tái tạo ở những nơi mask chỉ định,
    # và giữ lại phần ảnh gốc ở những nơi còn lại.
    # Đây là một phép toán alpha compositing cơ bản.
    final_output_tensor = restored_bg * mask + original_input_tensor * (1.0 - mask)

    # Bỏ đi chiều "batch" (từ B,C,H,W -> C,H,W) và chuyển tensor về CPU
    # để có thể chuyển đổi sang định dạng NumPy/PIL.
    output_tensor = final_output_tensor.squeeze(0).cpu()

    # Chuyển đổi thứ tự các chiều của tensor từ (C, H, W) của PyTorch
    # sang (H, W, C) là định dạng chuẩn của NumPy/PIL cho ảnh.
    output_numpy_rgb = output_tensor.permute(1, 2, 0).numpy()

    # Chuyển đổi (phản-chuẩn hóa) giá trị pixel từ khoảng [0.0, 1.0] trở lại
    # khoảng [0, 255] và đổi kiểu dữ liệu thành integer 8-bit không dấu (uint8).
    output_numpy_rgb = (output_numpy_rgb * 255).astype(np.uint8)

    # Tạo một đối tượng ảnh của Pillow từ mảng NumPy đã xử lý.
    output_image = Image.fromarray(output_numpy_rgb)

    # Lưu đối tượng ảnh này thành file tại đường dẫn đã cho.
    output_image.save(output_path)

def evaluate_metrics(ground_truth_dir, restoration_image_dir, device="cuda"):
    # Khởi tạo metric với data_range 
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = piq.LPIPS(replace_pooling=True).to(device)
    
    transform = transforms.ToTensor()
    
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    
    # Duyệt qua từng file ảnh
    for filename in tqdm(sorted(os.listdir(ground_truth_dir))):
        gt_path = os.path.join(ground_truth_dir, filename)
        rest_path = os.path.join(restoration_image_dir, filename)
        
        if not os.path.exists(rest_path):
            print(f"⚠️ Không tìm thấy ảnh khôi phục cho {filename}, bỏ qua.")
            continue
        
        # Đọc ảnh
        gt_img = Image.open(gt_path).convert("RGB")
        rest_img = Image.open(rest_path).convert("RGB")
        
        # Resize để cùng kích thước
        if gt_img.size != rest_img.size:
            rest_img = rest_img.resize(gt_img.size, Image.BICUBIC)
        
        # Tensor hóa
        gt_tensor = transform(gt_img).unsqueeze(0).to(device)  # [1, C, H, W]
        rest_tensor = transform(rest_img).unsqueeze(0).to(device)
        
        # Tính metric
        psnr_scores.append(psnr_metric(rest_tensor, gt_tensor).item())
        ssim_scores.append(ssim_metric(rest_tensor, gt_tensor).item())
        lpips_scores.append(lpips_metric(rest_tensor, gt_tensor).item())
    
    # Kết quả trung bình
    results = {
        "PSNR": sum(psnr_scores)/len(psnr_scores) if psnr_scores else None,
        "SSIM": sum(ssim_scores)/len(ssim_scores) if ssim_scores else None,
        "LPIPS": sum(lpips_scores)/len(lpips_scores) if lpips_scores else None,
    }
    return results

# --- Ví dụ chạy ---
if __name__ == "__main__":

    arg_param = argparse.ArgumentParser(description="MNetold Evaluation")
    arg_param.add_argument('-g', '--ground_truth_dir', type=str, required=True, help='Đường dẫn đến thư mục chứa ảnh ground truth.')
    arg_param.add_argument('-r', '--restoration_image_dir', type=str, required=True, help='Đường dẫn đến thư mục chứa ảnh khôi phục.')
    arg_param.add_argument('-w', '--weight', type=str, required=True, help='Đường dẫn đến file trọng số của model (.pth.tar).')
    arg_param.add_argument('-i', '--image', type=str, required=True, help='Đường dẫn đến thư mục chứa ảnh đầu vào.')
    
    # Parse arguments
    args = arg_param.parse_args()
    
    # Create output directory
    os.makedirs(args.restoration_image_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model configuration args
    model_args = argparse.Namespace(k1=5, k2=2, k3=1, srm=1, inf=1, crf=1, sharing=1)
    print(">> Khởi tạo kiến trúc model MNetold...")
    model = MNetold(args=model_args)

    print(f">> Sử dụng thiết bị tính toán: {device}")

    print(f">> Đang nạp trọng số model từ: {args.weight}")
    checkpoint = torch.load(args.weight, map_location=device)
    # Khối try-except này xử lý 2 cách lưu file checkpoint phổ biến
    try:
        # Cách 1: Trọng số nằm trong key 'state_dict'
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        # Cách 2: Trọng số được lưu trực tiếp
        model.load_state_dict(checkpoint)

    model.to(device)  
    model.eval()     
    print(f"Bắt đầu xóa tất cả watermark trong ảnh tại thư mục: {args.image}")
    images = os.listdir(args.image)
    for image in tqdm(images):
        image_path = os.path.join(args.image, image)
        input_tensor = preprocess_image(image_path, device)
        with torch.no_grad():
            restored_bg, mask = model(input_tensor)
        output_path = os.path.join(args.restoration_image_dir, image)
        postprocess_and_save(restored_bg, mask, input_tensor, output_path)
    
    print("Bắt đầu đánh giá chất lượng ảnh khôi phục...")
    scores = evaluate_metrics(args.ground_truth_dir, args.restoration_image_dir, device=device)

    # write results to json file
    with open("evaluation_results.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    print("Kết quả đánh giá:")
    for metric, score in scores.items():
        if score is not None:
            print(f"{metric}: {score:.4f}")

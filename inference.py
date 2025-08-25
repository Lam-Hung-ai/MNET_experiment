# ===================================================================
# 1. KHAI BÁO CÁC THƯ VIỆN CẦN THIẾT
# ===================================================================

import torch  # Thư viện nền tảng cho Deep Learning (PyTorch)
import numpy as np  # Thư viện để xử lý mảng số, đặc biệt hữu ích khi chuyển đổi giữa tensor và ảnh
import torchvision.transforms as transforms  # Cung cấp các hàm tiện ích để xử lý ảnh và chuyển đổi sang tensor
from argparse import Namespace, ArgumentParser  # Thư viện để xử lý các tham số đầu vào từ dòng lệnh
import os  # Thư viện để tương tác với hệ điều hành, ví dụ: tạo thư mục, lấy tên file
from PIL import Image  # Thư viện Pillow (PIL Fork) chuyên dùng để đọc, ghi và xử lý ảnh

# Import kiến trúc model từ file cục bộ
from model.MNetold import MNetold

# ===================================================================
# 2. HÀM TIỀN XỬ LÝ ẢNH ĐẦU VÀO
# ===================================================================
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
    print(f">> Bắt đầu tiền xử lý ảnh: {image_path}")

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

# ===================================================================
# 3. HÀM HẬU XỬ LÝ VÀ LƯU KẾT QUẢ
# ===================================================================
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
    print(">> Bắt đầu hậu xử lý và lưu ảnh...")

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
    print(f">> Thành công! Đã lưu ảnh kết quả tại: {output_path}")

# ===================================================================
# 4. HÀM ĐIỀU PHỐI CHÍNH
# ===================================================================
def run_inference_refactored(model_path, image_path, output_path):
    """
    Hàm chính, điều phối toàn bộ quá trình: nạp model, tiền xử lý,
    dự đoán (inference), và hậu xử lý.
    """
    # --- A. Cấu hình và nạp model ---
    args = Namespace(k1=5, k2=2, k3=1, srm=1, inf=1, crf=1, sharing=1)
    print(">> Khởi tạo kiến trúc model MNetold...")
    model = MNetold(args=args)

    # Tự động chọn GPU (cuda) nếu có, nếu không thì dùng CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">> Sử dụng thiết bị tính toán: {device}")

    print(f">> Đang nạp trọng số model từ: {model_path}")
    # Tải file checkpoint (chứa trọng số đã huấn luyện) vào bộ nhớ.
    checkpoint = torch.load(model_path, map_location=device)
    # Khối try-except này xử lý 2 cách lưu file checkpoint phổ biến
    try:
        # Cách 1: Trọng số nằm trong key 'state_dict'
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        # Cách 2: Trọng số được lưu trực tiếp
        model.load_state_dict(checkpoint)

    model.to(device)  # Chuyển toàn bộ model lên thiết bị đã chọn
    model.eval()      # **QUAN TRỌNG**: Chuyển model sang chế độ dự đoán (evaluation mode).
                      # Thao tác này sẽ vô hiệu hóa các layer như Dropout, BatchNorm.

    # --- B. Thực thi các bước ---

    # 1. Tiền xử lý ảnh đầu vào
    input_tensor = preprocess_image(image_path, device)

    # 2. Chạy dự đoán
    print(">> Bắt đầu quá trình dự đoán (inference)...")
    # torch.no_grad() giúp tiết kiệm bộ nhớ và tăng tốc độ vì không cần tính toán gradient
    with torch.no_grad():
        restored_bg, mask = model(input_tensor)

    # 3. Hậu xử lý và lưu kết quả
    postprocess_and_save(restored_bg, mask, input_tensor, output_path)

# ===================================================================
# 5. ĐIỂM BẮT ĐẦU THỰC THI SCRIPT
# ===================================================================
# Khối `if __name__ == '__main__':` đảm bảo code bên trong chỉ chạy
# khi bạn thực thi file này trực tiếp (vd: `python ten_file.py`),
# không chạy khi file này được import bởi một file khác.
if __name__ == '__main__':
    # --- A. Thiết lập trình phân tích tham số dòng lệnh ---
    # ArgumentParser giúp tạo ra một chương trình dòng lệnh thân thiện
    parser = ArgumentParser(description='Chạy inference cho model MNetold để xử lý ảnh.')

    # Định nghĩa các tham số mà chương trình sẽ nhận
    parser.add_argument('-i', '--image', type=str, required=True, help='Đường dẫn đến ảnh đầu vào.')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Đường dẫn đến file trọng số của model (.pth.tar).')

    # Đọc và phân tích các tham số người dùng nhập vào
    args = parser.parse_args()

    # --- B. Gán giá trị và tạo đường dẫn ---
    MODEL_CHECKPOINT = args.weights
    INPUT_IMAGE = args.image
    RESULTS_DIR = 'results'  # Tên thư mục để lưu kết quả

    # Tạo thư mục 'results' nếu nó chưa tồn tại
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Lấy tên file gốc từ đường dẫn input để đặt tên cho file output
    # Ví dụ: /path/to/my_photo.jpg -> my_photo.jpg
    image_name = os.path.basename(INPUT_IMAGE)

    # Tạo đường dẫn đầy đủ cho file output
    # Ví dụ: results/my_photo.jpg
    final_output_path = os.path.join(RESULTS_DIR, image_name)

    # --- C. Gọi hàm chính để bắt đầu thực thi ---
    run_inference_refactored(MODEL_CHECKPOINT, INPUT_IMAGE, final_output_path)
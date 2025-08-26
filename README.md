# MNet: Mạng Đa Quy Mô để loại bỏ Watermark

[Đây](https://github.com/Aitchson-Hwang/MNet) là mã nguồn cho bài báo [**"MNet: A multi-scale network for visible watermark removal"**](https://www.sciencedirect.com/science/article/abs/pii/S0893608024008906), một phương pháp tiên tiến để loại bỏ các hình mờ có thể nhìn thấy khỏi hình ảnh.



## 1. Giới thiệu 📝

Dự án này giải quyết bài toán loại bỏ hình mờ có thể nhìn thấy, một nhiệm vụ đầy thách thức do sự đa dạng về hình dạng, màu sắc và kết cấu của hình mờ. Các phương pháp trước đây thường gặp vấn đề về chất lượng khôi phục nền thấp hoặc tích tụ lỗi trong các kiến trúc nhiều giai đoạn.

MNet đề xuất một hướng tiếp cận mới: một **mạng một giai đoạn, đa quy mô** hiệu quả. Kiến trúc này giúp đơn giản hóa quá trình tối ưu hóa và tránh được các vấn đề trên, đồng thời tạo ra kết quả hình ảnh chất lượng vượt trội so với các phương pháp tiên tiến nhất (SOTA).

---

## 2. Kiến trúc MNet ⚙️

MNet là một mô hình học đa nhiệm, thực hiện đồng thời hai tác vụ chính thông qua hai nhánh xử lý song song:



* **Nhánh Khôi phục Nền (Background Restoration):** Thay vì dự đoán trực tiếp toàn bộ ảnh nền, MNet dự đoán một **"ảnh chống hình mờ" (anti-watermark image)**. Việc này giúp giảm độ khó của bài toán và cải thiện chất lượng khôi phục một cách đáng kể.
* **Nhánh Dự đoán Mặt nạ (Mask Prediction):** Xác định chính xác vị trí của hình mờ trên ảnh, giúp cho việc tái tạo được chính xác và giữ nguyên các vùng không bị ảnh hưởng.

Các đặc điểm kiến trúc cốt lõi bao gồm:
* **Kiến trúc Đa quy mô:** Sử dụng nhiều lớp **U-Nets** hoạt động trên các phiên bản ảnh có tỷ lệ khác nhau để nắm bắt cả chi tiết cục bộ và ngữ cảnh toàn cục.
* **Hợp nhất Đặc trưng (Feature Fusion):** Tích hợp các cơ chế hợp nhất đặc trưng **trong lớp (Intra-layer)** và **chéo lớp (Cross-layer)** để tăng cường luồng thông tin hiệu quả giữa các U-Nets.

---

## 3. Ưu điểm nổi bật ✨

* **Hiệu suất Vượt trội:** Đạt kết quả SOTA (State-of-the-art), vượt qua các phương pháp trước đó trên các bộ dữ liệu tiêu chuẩn (LOGO-H, LOGO-L, LOGO-Gray).
* **Chất lượng Hình ảnh Cao:** Tạo ra các hình ảnh được khôi phục sắc nét, rõ ràng và có chất lượng thị giác cao, ít bị mờ hay còn sót lại chi tiết thừa.
* **Kiến trúc Hiệu quả:** Chứng minh rằng một mạng một giai đoạn được thiết kế tốt có thể hiệu quả hơn các kiến trúc hai giai đoạn phức tạp.
* **Linh hoạt:** Kiến trúc cho phép thay đổi số lượng U-Nets trong mỗi lớp để tinh chỉnh hiệu suất và độ phức tạp.

Chắc chắn rồi\! Dưới đây là nội dung đã được cập nhật lại với chỉ mục bắt đầu từ mục số 4, giữ nguyên toàn bộ hướng dẫn chi tiết như trước.

-----

## 4\. Hướng dẫn cài đặt

Bạn có thể truy cập **web** để có thể trải nghiệm ngay.  
Hoặc có thể **chạy thủ công bằng CLI**.  
Hoặc có thể **thiết lập web để thao tác**.  

### 4.1 Chạy bằng CLI

Cách này phù hợp nếu bạn muốn kiểm tra nhanh khả năng của mô hình trên một ảnh.

#### **Bước 1: Cài đặt môi trường**

1.  **Clone repository** (Nếu bạn chưa có mã nguồn):

    ```bash
    git clone <your-repository-url>
    cd MNet_experiment
    ```

2.  **Cài đặt PyTorch:**
    Truy cập [trang chủ của PyTorch](https://pytorch.org/get-started/locally/) và làm theo hướng dẫn để cài đặt phiên bản phù hợp với hệ điều hành và phần cứng (CPU hoặc GPU) của bạn.

3.  **Cài đặt các thư viện phụ thuộc:**

    ```bash
    pip install -r requirements.txt
    ```

#### **Bước 2: Tải trọng số (Weight)**

Tải file trọng số của mô hình từ link sau và đặt vào thư mục `weight` ở gốc của dự án.

  * **Link tải:** [Google Drive](https://drive.google.com/drive/folders/1w54NjX69jYioTY8YYzAQasVhAlfI6x10?usp=drive_link)
  * **Đường dẫn lưu file:** `MNet_experiment/weight/model_best.pth.tar`

#### **Bước 3: Chạy suy luận (Inference)**

Sử dụng lệnh sau để chạy mô hình trên ảnh mẫu. Kết quả sẽ được lưu trong thư mục `results`.

```bash
python inference.py -w weight/model_best.pth.tar -i example_images/COCO_val2014_000000014338-Jules_Logo-175.png
```

-----

### 4.2 Thiết lập web để thao tác

Cách này sẽ khởi chạy một giao diện web cho phép bạn tải ảnh lên và nhận kết quả một cách trực quan.

#### **Bước 1: Chuẩn bị chung**

1.  Thực hiện lại **Bước 1** và **Bước 2** của phần **4.1 Chạy bằng CLI** để đảm bảo bạn đã có môi trường Python và file trọng số.
2.  **Quan trọng:** Sao chép file trọng số `model_best.pth.tar` từ thư mục `weight` ở gốc vào thư mục `web/backend/weight/`.

#### **Bước 2: Chạy Backend (Server)**

1.  Di chuyển vào thư mục `backend`:

    ```bash
    cd web/backend
    ```

2.  Khởi động server FastAPI bằng `uvicorn`. Server sẽ chạy tại địa chỉ `http://localhost:8000`.

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

    *Để terminal này chạy.*

#### **Bước 3: Chạy Frontend (Giao diện người dùng)**

1.  Mở một **terminal mới** và di chuyển vào thư mục `frontend`:

    ```bash
    cd web/frontend
    ```

2.  **Cài đặt các gói phụ thuộc** cho frontend (chỉ cần làm lần đầu):

    ```bash
    pnpm dev -H 0.0.0.0
    ```

3.  **Khởi động giao diện người dùng:**

    ```bash
    pnpm dev
    ```

4.  Mở trình duyệt và truy cập vào địa chỉ `http://localhost:3000` để bắt đầu trải nghiệm.
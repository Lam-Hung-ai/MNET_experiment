export default function FAQs() {
    return (
        <section className="scroll-py-16 py-16 md:scroll-py-32 md:py-32">
            <div className="mx-auto max-w-5xl px-6">
                <div className="grid gap-y-12 px-2 lg:[grid-template-columns:1fr_auto]">
                    <div className="text-center lg:text-left">
                        <h2 className="mb-4 text-3xl font-semibold md:text-4xl">
                            Những <br className="hidden lg:block" /> Câu Hỏi <br className="hidden lg:block" />
                            Thường Gặp
                        </h2>
                        <p>Bạn có thắc mắc gì không?</p>
                    </div>

                    <div className="divide-y divide-dashed sm:mx-auto sm:max-w-lg lg:mx-0">
                        <div className="pb-6">
                            <h3 className="font-medium">MNet là gì và nó giải quyết vấn đề gì?</h3>
                            <p className="text-muted-foreground mt-4">MNet là một mạng nơ-ron một giai đoạn, đa quy mô được thiết kế chuyên biệt để giải quyết bài toán loại bỏ hình mờ có thể nhìn thấy (visible watermark removal) khỏi hình ảnh. Mục tiêu của nó là tái tạo lại vùng ảnh bị che khuất bởi hình mờ, trả lại bức ảnh về trạng thái gốc một cách chân thực nhất có thể.</p>
                        </div>

                        <div className="py-6">
                            <h3 className="font-medium">MNet khác biệt như thế nào so với các phương pháp loại bỏ hình mờ khác?</h3>
                            <p className="text-muted-foreground mt-4">Điểm khác biệt chính của MNet nằm ở ba khía cạnh:</p>

                            <ol className="list-outside list-decimal space-y-2 pl-4">
                                <li className="text-muted-foreground mt-4"><strong>Kiến trúc một giai đoạn hiệu quả:</strong> Không giống các mô hình hai giai đoạn phức tạp có thể gây tích tụ lỗi, MNet là một mạng một giai đoạn được tối ưu hóa, giúp đơn giản hóa quá trình huấn luyện và tránh được các vấn đề trên.</li>
                                <li className="text-muted-foreground mt-4"><strong>Dự đoán "ảnh chống hình mờ":</strong> Thay vì cố gắng tái tạo toàn bộ nền ảnh, MNet dự đoán một "ảnh chống hình mờ" (anti-watermark image), là phần thông tin cần thiết để cộng vào ảnh gốc nhằm vô hiệu hóa hình mờ. Cách tiếp cận này giúp giảm độ khó của bài toán.</li>
                                <li className="text-muted-foreground mt-4"><strong>Cấu trúc đa quy mô và hợp nhất đặc trưng:</strong> MNet xử lý ảnh ở nhiều tỷ lệ khác nhau và sử dụng các cơ chế hợp nhất đặc trưng (chéo lớp và trong lớp) để luồng thông tin hiệu quả hơn, giúp tái tạo chi tiết tốt hơn so với các mạng một giai đoạn truyền thống.</li>
                            </ol>
                        </div>

                        <div className="py-6">
                            <h3 className="font-medium">Kiến trúc cốt lõi của MNet bao gồm những gì?</h3>
                            <p className="text-muted-foreground mt-4">MNet được xây dựng dựa trên các thành phần chính sau:</p>
                            <ul className="list-outside list-disc space-y-2 pl-4">
                                <li className="text-muted-foreground">Hai nhánh tác vụ: một nhánh để khôi phục nền (Background Restoration) và một nhánh để dự đoán mặt nạ (Mask Prediction) xác định vị trí hình mờ.</li>
                                <li className="text-muted-foreground">Cấu trúc 3 lớp: mỗi nhánh có 3 lớp, tương ứng với 3 tỷ lệ khác nhau của ảnh đầu vào.</li>
                                <li className="text-muted-foreground">Các khối U-Net xếp chồng: mỗi lớp bao gồm một chuỗi các khối U-Net đơn giản được xếp chồng lên nhau để liên tục tinh chỉnh các đặc trưng của ảnh.</li>
                            </ul>
                        </div>

                        <div className="py-6">
                            <h3 className="font-medium">U-Net được sử dụng trong MNet có phải là loại U-Net kinh điển không?</h3>
                            <p className="text-muted-foreground mt-4">Không hoàn toàn. U-Net trong MNet là một biến thể hiện đại thuộc họ ResUnet. Thay vì sử dụng các khối tích chập đơn giản như U-Net gốc, nó sử dụng các Khối Dư (Residual Blocks) làm đơn vị xây dựng chính, giúp huấn luyện các mạng sâu hiệu quả hơn và cải thiện chất lượng khôi phục ảnh.</p>
                        </div>

                        <div className="py-6">
                            <h3 className="font-medium">Tôi có thể tùy chỉnh độ phức tạp của model MNet không?</h3>
                            <p className="text-muted-foreground mt-4">Có. Một trong những ưu điểm của MNet là kiến trúc linh hoạt của nó. Bạn có thể thay đổi độ phức tạp và hiệu suất của model bằng cách điều chỉnh số lượng các khối U-Net được xếp chồng trong mỗi lớp thông qua các tham số k1, k2 và k3 khi khởi tạo model. Tăng số lượng U-Net ở lớp có độ phân giải cao nhất (k1) thường giúp cải thiện chất lượng hình ảnh.</p>
                        </div>

                        <div className="py-6">
                            <h3 className="font-medium">Làm thế nào để sử dụng MNet để xử lý một ảnh của riêng tôi?</h3>
                            <p className="text-muted-foreground mt-4">Quy trình chung:</p>
                            <ol className="list-outside list-decimal space-y-2 pl-4">
                                <li className="text-muted-foreground">Khởi tạo model MNet với các tham số (k1, k2, k3, ...).</li>
                                <li className="text-muted-foreground">Nạp trọng số đã huấn luyện từ file .pth.tar.</li>
                                <li className="text-muted-foreground">Chuẩn bị ảnh đầu vào: đọc ảnh (ví dụ OpenCV), chuyển sang RGB, resize về 256×256 và chuyển thành tensor.</li>
                                <li className="text-muted-foreground">Chạy inference: đưa tensor qua model để nhận ảnh nền đã khôi phục và mặt nạ.</li>
                                <li className="text-muted-foreground">Hậu xử lý: kết hợp kết quả model với ảnh gốc để tạo ảnh cuối cùng và lưu ra file.</li>
                            </ol>
                            <p className="text-muted-foreground mt-4">Đây chỉ là mô tả tổng quan; ví dụ mã Python cụ thể sẽ bao gồm các bước khởi tạo model, tải trọng số, xử lý ảnh với OpenCV/NumPy, chuyển sang PyTorch tensor, inference và lưu kết quả.</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}

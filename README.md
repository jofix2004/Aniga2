# Aniga - Too easy, not fun.

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 🚀 Google Colab

[![Mở trong Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uwWlg9bsheH25_C-q1iXdzJDKmREwSEy?usp=sharing)

## 💻 Local

### Yêu cầu tiên quyết

-   [Python](https://www.python.org/downloads/) (phiên bản 3.10 trở lên)
-   [Git](https://git-scm.com/downloads/)
-   Card đồ họa NVIDIA với CUDA được cài đặt (khuyến nghị mạnh mẽ để có hiệu năng tốt nhất)

### Các bước cài đặt

**1. Clone Repository**

Mở terminal hoặc command prompt và chạy lệnh sau:
```bash
git clone https://github.com/jofix2004/Aniga2.git
cd Aniga2
```

**2. Tạo và Kích hoạt Môi trường ảo (Khuyến nghị)**

Việc sử dụng môi trường ảo giúp tránh xung đột thư viện.
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

**3. Cài đặt các Thư viện**

Quy trình cài đặt được chia làm hai bước để đảm bảo tương thích phần cứng.

**Bước 3a: Cài đặt PyTorch**

Truy cập [trang web chính thức của PyTorch](https://pytorch.org/get-started/locally/) để lấy lệnh cài đặt chính xác nhất cho hệ thống của bạn (CUDA, CPU, OS).

*Ví dụ cho hệ thống có CUDA 12.1:*
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Bước 3b: Cài đặt các thư viện còn lại**

Sau khi PyTorch đã được cài đặt, hãy chạy lệnh sau để cài đặt tất cả các gói cần thiết khác:
```bash
pip install -r requirements.txt
```

**4. Chạy Ứng dụng**

Khi tất cả các thư viện đã được cài đặt, khởi động ứng dụng bằng lệnh:
```bash
python app.py
```

**5. Truy cập Giao diện**

Mở trình duyệt của bạn và truy cập vào địa chỉ [http://127.0.0.1:7860](http://127.0.0.1:7860).

## ⚙️ Cấu hình

Bạn có thể tùy chỉnh sâu hơn các logic của pipeline hợp nhất bằng cách chỉnh sửa file `ensemble_config.json`. File này cho phép bạn:
-   Định nghĩa các nhóm lớp (`class_groups`) để áp dụng các quy tắc lọc riêng.
-   Bật/tắt logic lọc trong nội bộ nhóm.
-   Thay đổi các giá trị mặc định cho các ngưỡng xử lý.
-   **Xử lý hình ảnh:** OpenCV
-   **Tăng tốc hiệu năng:** Numba
-   **OCR:** python-doctr, manga-ocr````

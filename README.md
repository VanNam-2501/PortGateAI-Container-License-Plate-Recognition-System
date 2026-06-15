# Smart Port Gate: E2E ALPR & Container OCR System

Hệ thống nhận diện biển số xe (ALPR) và mã container tự động dạng End-to-End (E2E) phục vụ quản lý phương tiện ra vào cổng cảng thông minh. Hệ thống tối ưu hóa luồng xử lý thời gian thực từ camera stream, tự động cắt, nắn phẳng ảnh và trích xuất thông tin chính xác bằng mô hình học máy.

## 🎥 Video Demo

<p align="center">
  <video src="demo.mp4" width="100%" height="auto" controls loop autoplay muted></video>
</p>

---

## 🚀 Tính năng nổi bật

- **Nhận diện biển số xe (ALPR)**: Hỗ trợ đa dạng các loại biển số Việt Nam bao gồm biển dân sự, biển quân sự, và biển ngoại giao. Tự động định dạng chuẩn (`59A-123.45`, `KP-12-34`, `29-NG-123-45`).
- **OCR Mã Container**: Trích xuất chính xác mã container theo tiêu chuẩn ISO 6346 (bao gồm 4 ký tự chữ sở hữu, 6 ký tự số sê-ri, và 1 ký tự kiểm tra).
- **Thuật toán Tự sửa lỗi OCR**: Nhận dạng vị trí ký tự để tự động sửa các lỗi nhầm lẫn phổ biến (ví dụ: `0` thành `O` ở vùng chữ, hoặc ngược lại ở vùng số).
- **Kiểm tra Check-Digit ISO 6346**: Xác thực thuật toán tổng kiểm tra (Check-digit) để giảm thiểu tối đa sai sót nhận diện.
- **Nắn phẳng phối cảnh (Perspective Warp)**: Tự động sắp xếp các đỉnh hộp xoay (OBB) từ YOLO theo chiều kim đồng hồ và đưa vùng nghiêng về góc nhìn phẳng 90 độ, tối ưu hóa độ chính xác cho OCR.
- **Gom nhóm đa khung hình (Multi-frame Aggregation)**: Thuật toán bầu chọn đa số (Majority Voting) trên chuỗi frame video giúp nâng cao độ chính xác khi xe di chuyển.
- **Giao diện Dashboard trực quan**: Theo dõi thời gian thực kết quả nhận diện, trạng thái cổng, và thời gian trễ (latency) của từng luồng xử lý.

---

## 🛠️ Công nghệ sử dụng

- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Deep Learning / AI**:
  - **Object Detection**: YOLOv11-OBB (định vị vùng biển số & mã vách container nghiêng)
  - **OCR Engine**: PaddleOCR (trích xuất ký tự) / ONNX Runtime
- **Image Processing**: OpenCV (Warp perspective, Image quality thresholding)
- **Frontend / Dashboard**: HTML5, TailwindCSS (Vanilla setup), JavaScript ES6 (Server-Sent Events)
- **Testing**: Pytest, Pytest-cov

---

## 📂 Cấu trúc Thư mục Dự án

```text
container-ocr/
├── app/
│   ├── main.py                # Điểm khởi chạy FastAPI (FastAPI entrypoint)
│   ├── config.py              # Cấu hình đường dẫn và biến môi trường
│   ├── schemas.py             # Định nghĩa schemas Pydantic cho dữ liệu đầu vào/ra
│   ├── detectors/             # Module bọc mô hình YOLOv11-OBB
│   ├── ocr/                   # Module tích hợp các engine OCR (PaddleOCR, ONNX, Mock)
│   ├── pipeline/              # Bộ điều phối (Orchestrator) luồng xử lý 5 tầng
│   ├── static/                # Giao diện web dashboard theo dõi thời gian thực
│   └── utils/                 # Các tiện ích xử lý ảnh, validators và simulator
├── config/                    # Thư mục cấu hình (settings.yaml, roi_config.json)
├── models/                    # Lưu trữ các file trọng số mô hình (.pt, .onnx)
├── data/
│   └── uploads/               # Nơi lưu trữ video/ảnh đầu vào lúc runtime
├── outputs/
│   └── results/               # Chứa ảnh cắt biển số/container phục vụ dashboard
├── scripts/                   # Các script huấn luyện, kiểm định và benchmark
├── tests/                     # Hệ thống kiểm thử tự động (Unit / E2E Tests)
├── .env                       # Cấu hình cục bộ cho runtime
├── requirements.txt           # Danh sách thư viện Python cần thiết
└── run_inference.sh           # Script khởi chạy nhanh luồng test inference
```

---

## 💻 Hướng dẫn Cài đặt & Chạy nhanh

### 1. Cài đặt Môi trường
Yêu cầu hệ thống đã cài đặt Python 3.8+ và `venv`.

```bash
# Khởi tạo và kích hoạt môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Cập nhật pip và cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Thiết lập Cấu hình Môi trường
Sao chép cấu hình mẫu từ `.env.example`:
```bash
cp .env.example .env
```
*(Chỉnh sửa các tham số trong `.env` để phù hợp với môi trường chạy thực tế của bạn)*

### 3. Chạy Server Giao diện Dashboard (Web)
Khởi chạy ứng dụng bằng Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Sau đó, truy cập giao diện giám sát tại địa chỉ: `http://localhost:8000`.

### 4. Kiểm thử hệ thống (Inference & Tests)
Chạy bộ kiểm thử tự động để kiểm tra logic xử lý:
```bash
PYTHONPATH=. pytest tests -v
```

Để chạy nhanh chương trình thử nghiệm nhận diện trên ảnh mẫu:
```bash
./run_inference.sh
```

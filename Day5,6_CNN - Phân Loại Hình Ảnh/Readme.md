# 📌 Dự án: Phân loại ảnh với CNN

## 📝 Giới thiệu

Dự án này sử dụng **Mạng nơ-ron tích chập (CNN - Convolutional Neural Network)** để phân loại ảnh. CNN là một mô hình mạnh mẽ giúp nhận diện đặc trưng hình ảnh và được ứng dụng rộng rãi trong thị giác máy tính.

## 📂 Cấu trúc thư mục

```
📁 CNN_Image_Classification
│-- 📄 CNN.ipynb       # Notebook chính chứa toàn bộ quá trình xử lý dữ liệu và xây dựng mô hình
│-- 📄 README.md        # Tóm lược về dự án
│-- 📁 dataset/        # Thư mục chứa ảnh đầu vào
```

## 🔧 Công nghệ sử dụng

- **Python** (ngôn ngữ lập trình chính)
- **TensorFlow & Keras** (xây dựng mô hình CNN)
- **Matplotlib & Seaborn** (trực quan hóa dữ liệu)
- **OpenCV & PIL** (xử lý ảnh)

## 📊 Các bước thực hiện

1. **Tiền xử lý dữ liệu**:
   - Đọc và chuẩn bị dữ liệu ảnh.
   - Chia dữ liệu thành tập train và test.
   - Chuẩn hóa dữ liệu để tối ưu mô hình.
2. **Xây dựng mô hình CNN**:
   - Thiết kế các lớp tích chập (Convolutional layers).
   - Thêm các lớp kích hoạt ReLU, MaxPooling.
   - Thêm các lớp Dense để tạo đầu ra.
3. **Huấn luyện mô hình**:
   - Dùng `adam` optimizer và `categorical_crossentropy` làm hàm mất mát.
   - Sử dụng tập validation để theo dõi hiệu suất mô hình.
4. **Đánh giá mô hình**:
   - Hiển thị độ chính xác và hàm mất mát.
   - Trực quan hóa dự đoán bằng biểu đồ.

## 📈 Kết quả đạt được

- Mô hình có khả năng phân loại ảnh với độ chính xác cao.
- Biểu đồ trực quan giúp theo dõi quá trình huấn luyện.

## 🚀 Cách chạy dự án

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install numpy pandas tensorflow matplotlib opencv-python pillow
   ```
2. Chạy notebook `CNN.ipynb` để thực hiện từng bước huấn luyện.
3. Kiểm tra kết quả dự đoán trên tập test.

## 📌 Hướng phát triển

- Cải thiện mô hình bằng cách thêm Dropout để tránh overfitting.
- Sử dụng tập dữ liệu lớn hơn để tăng độ chính xác.
- Thử nghiệm với các mô hình CNN tiên tiến như VGG16, ResNet.

📩 Nếu có câu hỏi, hãy liên hệ hoặc để lại issue! 🚀

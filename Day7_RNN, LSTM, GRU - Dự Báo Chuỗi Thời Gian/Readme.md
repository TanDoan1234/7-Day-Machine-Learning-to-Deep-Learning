# 📌 Dự án: Dự đoán chuỗi thời gian với LSTM

## 📝 Giới thiệu

Dự án này sử dụng mạng **LSTM (Long Short-Term Memory)** để dự đoán chuỗi thời gian dựa trên dữ liệu quá khứ. LSTM là một loại mạng **Recurrent Neural Network (RNN)** đặc biệt mạnh mẽ trong việc xử lý dữ liệu tuần tự.

## 📂 Cấu trúc thư mục

```
📁 LSTM_Project
│-- 📄 LSTM.ipynb        # Notebook chính với toàn bộ quá trình xử lý dữ liệu và huấn luyện mô hình
│-- 📄 LSTM_explanation.md  # File Markdown giải thích từng phần trong code
│-- 📄 README.md        # Tóm lược về dự án
│-- 📄 data.csv         # Dữ liệu đầu vào cho mô hình
```

## 🔧 Công nghệ sử dụng

- **Python** (ngôn ngữ lập trình chính)
- **TensorFlow/Keras** (xây dựng mô hình LSTM)
- **NumPy & Pandas** (xử lý dữ liệu)
- **Matplotlib** (vẽ biểu đồ)
- **Scikit-learn** (chuẩn hóa dữ liệu)

## 📊 Quy trình thực hiện

1. **Import thư viện** cần thiết.
2. **Load và tiền xử lý dữ liệu**:
   - Chuẩn hóa dữ liệu bằng `MinMaxScaler`.
   - Chia dữ liệu thành tập **huấn luyện** và **kiểm tra**.
3. **Chuẩn bị dữ liệu cho mô hình LSTM**:
   - Chia dữ liệu thành các **chuỗi thời gian** phù hợp với LSTM.
4. **Xây dựng mô hình LSTM**:
   - Sử dụng 2 tầng LSTM và 1 tầng Dense để dự đoán giá trị tiếp theo.
5. **Huấn luyện mô hình**:
   - Dùng **Adam Optimizer** và **Mean Squared Error (MSE)** làm hàm mất mát.
   - Theo dõi quá trình giảm lỗi bằng biểu đồ.
6. **Dự đoán và đánh giá**:
   - Sử dụng tập kiểm tra để đánh giá mô hình.
   - Chuyển kết quả về giá trị ban đầu bằng `inverse_transform()`.

## 📈 Kết quả đạt được

- Mô hình có thể dự đoán xu hướng của dữ liệu thời gian.
- Biểu đồ loss giúp kiểm tra độ hội tụ của mô hình.

## 🚀 Cách chạy dự án

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install numpy pandas tensorflow scikit-learn matplotlib
   ```
2. Chạy notebook `LSTM.ipynb` để thực hiện từng bước.
3. Kiểm tra kết quả dự đoán và biểu đồ loss.

## 📌 Ghi chú

- Có thể tinh chỉnh `seq_length`, số **epochs**, hoặc cấu trúc LSTM để cải thiện độ chính xác.
- Dữ liệu cần được chuẩn bị cẩn thận trước khi đưa vào mô hình.

📩 Nếu có câu hỏi, hãy liên hệ hoặc để lại issue! 🚀

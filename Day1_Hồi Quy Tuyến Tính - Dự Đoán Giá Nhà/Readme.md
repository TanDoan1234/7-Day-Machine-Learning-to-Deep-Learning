# 📌 Dự án: Dự đoán giá nhà với Machine Learning

## 📝 Giới thiệu

Dự án này sử dụng các thuật toán Machine Learning để dự đoán giá nhà dựa trên các đặc trưng về bất động sản. Tập dữ liệu được sử dụng là **Boston Housing Dataset**, một bộ dữ liệu phổ biến trong phân tích giá nhà.

## 📂 Cấu trúc thư mục

```
📁 House_Price_Prediction
│-- 📄 test.ipynb        # Notebook chính với quá trình xử lý dữ liệu và huấn luyện mô hình
│-- 📄 README.md         # Tóm lược về dự án
|-- 📄 Linear_explain.md # Maskdown giải thích từng phần code
│-- 📄 housing.csv       # Dữ liệu đầu vào cho mô hình
```

## 📊 Dataset

Bộ dữ liệu gồm nhiều thông tin về các đặc điểm của bất động sản. Dưới đây là một số cột quan trọng:
| **Tên cột** | **Ý nghĩa** |
|:------------:|:-----------:|
| **CRIM** | Tỷ lệ tội phạm trên đầu người theo từng thị trấn. |
| **ZN** | Tỷ lệ đất dân cư được quy hoạch cho các lô đất trên 25.000 ft². |
| **INDUS** | Tỷ lệ phần trăm diện tích đất dành cho hoạt động kinh doanh phi bán lẻ. |
| **CHAS** | Biến giả (1 nếu thị trấn giáp sông Charles; 0 nếu không). |
| **NOX** | Nồng độ oxit nitric. |
| **RM** | Số phòng trung bình trên mỗi căn nhà. |
| **AGE** | Tỷ lệ nhà được xây trước năm 1940. |
| **DIS** | Khoảng cách trung bình đến các trung tâm việc làm. |
| **RAD** | Chỉ số tiếp cận đường cao tốc. |
| **TAX** | Mức thuế bất động sản. |
| **PTRATIO** | Tỷ lệ học sinh/giáo viên theo thị trấn. |
| **LSTAT** | Tỷ lệ dân số thuộc nhóm thu nhập thấp. |
| **MEDV** | Giá trị trung vị của nhà (đơn vị: $1.000). |

## 🔧 Công nghệ sử dụng

- **Python** (ngôn ngữ lập trình chính)
- **Scikit-learn** (các thuật toán Machine Learning)
- **Pandas & NumPy** (xử lý dữ liệu)
- **Matplotlib & Seaborn** (trực quan hóa dữ liệu)

## 📊 Các bước thực hiện

1. **Tiền xử lý dữ liệu**:
   - Xử lý giá trị khuyết thiếu (nếu có).
   - Chuẩn hóa dữ liệu bằng `StandardScaler`.
   - Chia dữ liệu thành tập **train** và **test** (80/20).
2. **Xây dựng mô hình**:
   - Thử nghiệm các mô hình như **Linear Regression, Decision Tree, Random Forest, XGBoost, và MLP**.
   - Điều chỉnh tham số và chọn mô hình tối ưu nhất.
3. **Đánh giá mô hình**:
   - Sử dụng các chỉ số như **MSE (Mean Squared Error), RMSE (Root Mean Squared Error), R² Score**.
   - So sánh giá trị thực tế và giá trị dự đoán bằng biểu đồ.

## 📈 Kết quả đạt được

- Mô hình có thể dự đoán giá nhà với độ chính xác cao.
- Biểu đồ trực quan giúp phân tích sự khác biệt giữa giá thực tế và giá dự đoán.

## 🚀 Cách chạy dự án

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
2. Chạy notebook `test.ipynb` để thực hiện từng bước.
3. Kiểm tra kết quả dự đoán và trực quan hóa dữ liệu.

## 📌 Hướng phát triển

- Điều chỉnh các **hyperparameters** để cải thiện độ chính xác.
- Thử nghiệm với các thuật toán khác như **Neural Networks, Gradient Boosting**.
- Thu thập thêm dữ liệu thực tế để mở rộng mô hình.

📩 Nếu có câu hỏi, hãy liên hệ hoặc để lại issue! 🚀

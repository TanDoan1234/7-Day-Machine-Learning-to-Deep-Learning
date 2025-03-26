# 📌 Dự án: Phân tích và dự đoán rủi ro tín dụng

## 📝 Giới thiệu

Dự án này nhằm phân tích dữ liệu tài chính và xây dựng mô hình dự đoán rủi ro tín dụng của khách hàng dựa trên các đặc điểm cá nhân và tài chính. Mô hình sẽ giúp đánh giá khả năng **vỡ nợ (bad risk)** hoặc **không vỡ nợ (good risk)** của một khách hàng khi vay vốn.

## 📂 Cấu trúc thư mục

```
📁 Credit_Risk_Analysis
│-- 📄 test.ipynb              # Notebook chính chứa toàn bộ quá trình xử lý dữ liệu và xây dựng mô hình
│-- 📄 README.md               # Tóm lược về dự án
|-- 📄 Credit_risk.md          # Maskdown giải thích từng phần code
│-- 📄 german_credit_data.csv  # Dữ liệu khách hàng dùng để phân tích
```

## 📊 Dataset

Bộ dữ liệu chứa thông tin về khách hàng, bao gồm các đặc điểm như tuổi, thu nhập, tình trạng tài chính, khoản vay và mục đích vay. Một số cột quan trọng:

| **Tên biến**         | **Ý nghĩa**                                                 |
| -------------------- | ----------------------------------------------------------- |
| **Age**              | Tuổi của khách hàng.                                        |
| **Sex**              | Giới tính (male/female).                                    |
| **Job**              | Loại công việc (số nguyên).                                 |
| **Housing**          | Tình trạng nhà ở (rent/own/free).                           |
| **Saving accounts**  | Số dư tài khoản tiết kiệm (ít, vừa, nhiều hoặc NaN).        |
| **Checking account** | Số dư tài khoản thanh toán (little/moderate/rich hoặc NaN). |
| **Credit amount**    | Số tiền vay.                                                |
| **Duration**         | Thời gian vay (tháng).                                      |
| **Purpose**          | Mục đích vay tiền (car, education, furniture, etc.).        |
| **Risk**             | Nhãn dự đoán (good = không vỡ nợ, bad = vỡ nợ).             |

## 🔧 Công nghệ sử dụng

- **Python** (ngôn ngữ lập trình chính)
- **Pandas & NumPy** (xử lý dữ liệu)
- **Matplotlib & Seaborn** (trực quan hóa dữ liệu)
- **Scikit-learn** (học máy, phân tích dữ liệu)

## 📊 Các bước thực hiện

1. **Tiền xử lý dữ liệu**:
   - Xử lý giá trị khuyết thiếu.
   - Biến đổi dữ liệu danh mục thành dạng số.
   - Chuẩn hóa dữ liệu bằng `StandardScaler`.
2. **Phân tích dữ liệu**:
   - Trực quan hóa phân bố các biến số.
   - Phân tích mối quan hệ giữa các đặc trưng và rủi ro tín dụng.
3. **Xây dựng mô hình Machine Learning**:
   - Chia dữ liệu thành tập train/test (80/20).
   - Thử nghiệm các mô hình như **Logistic Regression, Decision Tree, Random Forest, XGBoost**.
   - Điều chỉnh tham số và chọn mô hình tốt nhất.
4. **Đánh giá mô hình**:
   - Sử dụng các chỉ số như **Accuracy, Precision, Recall, F1-score, ROC-AUC** để đánh giá.
   - So sánh kết quả dự đoán với dữ liệu thực tế.

## 📈 Kết quả đạt được

- Mô hình có thể xác định khách hàng có nguy cơ vỡ nợ cao dựa trên dữ liệu tài chính cá nhân.
- Biểu đồ phân tích giúp phát hiện các yếu tố có ảnh hưởng lớn đến rủi ro tín dụng.

## 🚀 Cách chạy dự án

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
2. Chạy notebook `test.ipynb` để thực hiện từng bước.
3. Kiểm tra kết quả dự đoán và trực quan hóa dữ liệu.

## 📌 Hướng phát triển

- Điều chỉnh **hyperparameters** để cải thiện độ chính xác của mô hình.
- Thử nghiệm với các mô hình khác như **Neural Networks, Gradient Boosting**.
- Áp dụng trên các bộ dữ liệu thực tế của ngân hàng để kiểm tra độ hiệu quả.

📩 Nếu có câu hỏi, hãy liên hệ hoặc để lại issue! 🚀

# 📌 Dự án: Phân tích khách hàng với Machine Learning

## 📝 Giới thiệu

Dự án này sử dụng thuật toán **Phân cụm (Clustering)** để phân tích tập dữ liệu khách hàng dựa trên các đặc điểm như tuổi, thu nhập và chi tiêu. Kết quả giúp doanh nghiệp xác định nhóm khách hàng tiềm năng để tối ưu chiến lược tiếp thị.

## 📂 Cấu trúc thư mục

```
📁 Customer_Analysis
│-- 📄 test1.ipynb       # Notebook chính chứa toàn bộ quá trình xử lý dữ liệu và xây dựng mô hình
│-- 📄 README.md        # Tóm lược về dự án
│-- 📄 Mall_Customers.csv  # Dữ liệu khách hàng dùng để phân tích
```

## 📊 Dataset

Bộ dữ liệu **Mall_Customers.csv** chứa thông tin về khách hàng trung tâm mua sắm:

| **Tên cột**        | **Ý nghĩa**                  |
| ------------------ | ---------------------------- |
| **CustomerID**     | Mã khách hàng                |
| **Genre**          | Giới tính (Male/Female)      |
| **Age**            | Tuổi khách hàng              |
| **Annual Income**  | Thu nhập hàng năm (ngàn USD) |
| **Spending Score** | Điểm chi tiêu của khách hàng |

## 🔧 Công nghệ sử dụng

- **Python** (ngôn ngữ lập trình chính)
- **Pandas & NumPy** (xử lý dữ liệu)
- **Matplotlib & Seaborn** (trực quan hóa dữ liệu)
- **Scikit-learn** (thuật toán phân cụm K-Means)

## 📊 Các bước thực hiện

1. **Tiền xử lý dữ liệu**:
   - Kiểm tra dữ liệu và xử lý giá trị thiếu (nếu có).
   - Chuẩn hóa dữ liệu nếu cần.
2. **Phân tích dữ liệu**:
   - Thống kê các đặc điểm của khách hàng.
   - Trực quan hóa dữ liệu bằng biểu đồ.
3. **Phân cụm khách hàng bằng K-Means**:
   - Xác định số cụm tối ưu bằng phương pháp **Elbow Method**.
   - Áp dụng thuật toán **K-Means** để nhóm khách hàng.
   - Đánh giá kết quả phân cụm.
4. **Trực quan hóa kết quả**:
   - Hiển thị các cụm khách hàng bằng biểu đồ Scatter.

## 📈 Kết quả đạt được

- Phân loại khách hàng thành các nhóm riêng biệt dựa trên hành vi chi tiêu và thu nhập.
- Giúp doanh nghiệp xây dựng chiến lược tiếp cận khách hàng phù hợp hơn.

## 🚀 Cách chạy dự án

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
2. Chạy notebook `test1.ipynb` để thực hiện từng bước phân tích.
3. Quan sát kết quả phân cụm và trực quan hóa dữ liệu.

## 📌 Hướng phát triển

- Thử nghiệm với các thuật toán phân cụm khác như **DBSCAN, Agglomerative Clustering**.
- Kết hợp thêm dữ liệu khác để tăng độ chính xác của mô hình.
- Ứng dụng mô hình vào thực tế để cải thiện chiến lược tiếp thị.

📩 Nếu có câu hỏi, hãy liên hệ hoặc để lại issue! 🚀

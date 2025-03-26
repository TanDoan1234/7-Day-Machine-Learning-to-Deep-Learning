# Dự đoán doanh số bán hàng với MLP

## Giới thiệu

Dự án này nhằm dự đoán doanh số bán hàng dựa trên dữ liệu lịch sử sử dụng một mô hình MLP (Multilayer Perceptron). Dữ liệu bao gồm thông tin về thời gian, địa điểm và sản phẩm được bán.

## Dataset

Dataset được sử dụng có tên **annualSales2019.csv**, bao gồm 186,850 dòng với 6 cột:

- `Order ID`: Mã đơn hàng.
- `Product`: Tên sản phẩm.
- `Quantity Ordered`: Số lượng sản phẩm được đặt.
- `Price Each`: Giá mỗi sản phẩm.
- `Order Date`: Ngày và giờ đặt hàng.
- `Purchase Address`: Địa chỉ mua hàng.

## Các bước thực hiện

1. **Tiền xử lý dữ liệu**:

   - Xóa dữ liệu khuyết, xử lý lỗi dữ liệu.
   - Trích xuất thông tin quan trọng (tháng, giờ, ngày trong tuần, thành phố).
   - Chuẩn hóa dữ liệu bằng StandardScaler.

2. **Chia dữ liệu**:

   - Tách tập dữ liệu thành train/test theo tỉ lệ 80/20.

3. **Xây dựng mô hình MLP**:

   - Sử dụng TensorFlow/Keras.
   - Cấu trúc gồm nhiều tầng fully connected với hàm activation ReLU.
   - Sử dụng optimizer Adam và loss function MSE.

4. **Huấn luyện và đánh giá**:
   - Huấn luyện mô hình trong nhiều epoch và theo dõi loss.
   - Kiểm tra kết quả trên tập test.

## Kết quả

Mô hình có khả năng dự đoán doanh số bán hàng từ các biến đầu vào với độ chính xác khá. Có thể tối ưu thêm bằng tuning hyperparameter.

## Cách chạy

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
   ```
2. Chạy notebook hoặc script python.

## Liên hệ

Mọi đóng góp hoặc thắc mắc xin vui lòng liên hệ qua GitHub hoặc email.

# Giải thích từng phần của mã nguồn trong test.ipynb

## 1. Import thư viện

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

**Giải thích:**

- `numpy` và `pandas`: Xử lý dữ liệu dạng số và bảng.
- `matplotlib.pyplot` và `seaborn`: Vẽ biểu đồ để trực quan hóa dữ liệu.
- `train_test_split`: Chia dữ liệu thành tập train và test.
- `StandardScaler`: Chuẩn hóa dữ liệu để mô hình hoạt động tốt hơn.
- `LinearRegression`: Thuật toán hồi quy tuyến tính để dự đoán giá nhà.
- `mean_squared_error`, `r2_score`: Đánh giá hiệu suất của mô hình.

## 2. Đọc và xem xét dữ liệu

```python
df = pd.read_csv("data.csv")
print(df.head())
print(df.info())
print(df.describe())
```

**Giải thích:**

- `pd.read_csv("data.csv")`: Đọc dữ liệu từ file CSV.
- `df.head()`: Hiển thị 5 dòng đầu tiên.
- `df.info()`: Thông tin tổng quan về kiểu dữ liệu.
- `df.describe()`: Thống kê các giá trị trong dataset.

## 3. Xử lý dữ liệu

```python
df = df.dropna()
X = df.drop(columns=["MEDV"])
y = df["MEDV"]
```

**Giải thích:**

- `dropna()`: Loại bỏ các dòng có giá trị khuyết thiếu.
- `X = df.drop(columns=["MEDV"])`: Chọn các đặc trưng đầu vào (không bao gồm cột giá nhà `MEDV`).
- `y = df["MEDV"]`: Biến mục tiêu cần dự đoán.

## 4. Chia dữ liệu train/test

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Giải thích:**

- `test_size=0.2`: 20% dữ liệu dành cho kiểm tra, 80% để huấn luyện.
- `random_state=42`: Đảm bảo kết quả có thể tái lập.

## 5. Chuẩn hóa dữ liệu

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Giải thích:**

- `StandardScaler()`: Chuẩn hóa dữ liệu về phân phối chuẩn.
- `fit_transform(X_train)`: Áp dụng scaler lên tập train.
- `transform(X_test)`: Dùng cùng scaler để chuẩn hóa tập test.

## 6. Xây dựng mô hình Linear Regression

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

**Giải thích:**

- `LinearRegression()`: Khởi tạo mô hình hồi quy tuyến tính.
- `fit(X_train, y_train)`: Huấn luyện mô hình trên tập train.

## 7. Dự đoán

```python
y_pred = model.predict(X_test)
```

**Giải thích:**

- `predict(X_test)`: Dự đoán giá nhà trên tập test.

## 8. Đánh giá mô hình

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R² Score: {r2}')
```

**Giải thích:**

- `mean_squared_error(y_test, y_pred)`: Tính sai số trung bình bình phương (MSE).
- `r2_score(y_test, y_pred)`: Tính hệ số xác định R².
- `print(f'MSE: {mse}')`: Hiển thị kết quả lỗi.
- `print(f'R² Score: {r2}')`: Hiển thị độ chính xác của mô hình.

---

📌 **Lưu ý:** Nếu bạn cần thêm giải thích hoặc cải thiện mô hình, hãy liên hệ tôi nhé! 🚀

# Giải thích từng phần của mã nguồn trong test.ipynb

## 1. Import thư viện

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**Giải thích:**

- `numpy` và `pandas`: Xử lý dữ liệu dạng số và bảng.
- `matplotlib.pyplot` và `seaborn`: Vẽ biểu đồ để trực quan hóa dữ liệu.
- `train_test_split`: Chia dữ liệu thành tập train và test.
- `StandardScaler`: Chuẩn hóa dữ liệu để mô hình hoạt động tốt hơn.
- `LabelEncoder`: Chuyển đổi dữ liệu dạng danh mục thành số.
- `RandomForestClassifier`: Thuật toán phân loại sử dụng rừng ngẫu nhiên.
- `accuracy_score`, `classification_report`, `confusion_matrix`: Đánh giá hiệu suất của mô hình.

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
label_encoder = LabelEncoder()
df['Risk'] = label_encoder.fit_transform(df['Risk'])
```

**Giải thích:**

- `dropna()`: Loại bỏ các dòng có giá trị khuyết thiếu.
- `LabelEncoder()`: Chuyển đổi biến phân loại `Risk` thành dạng số (0: Good, 1: Bad).

## 4. Tách đặc trưng và nhãn

```python
X = df.drop(columns=["Risk"])
y = df["Risk"]
```

**Giải thích:**

- `X`: Chứa tất cả các cột trừ `Risk` (đặc trưng đầu vào).
- `y`: Cột `Risk` làm biến mục tiêu cần dự đoán.

## 5. Chia dữ liệu train/test

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Giải thích:**

- `test_size=0.2`: 20% dữ liệu dành cho kiểm tra, 80% để huấn luyện.
- `random_state=42`: Đảm bảo kết quả có thể tái lập.

## 6. Chuẩn hóa dữ liệu

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Giải thích:**

- `StandardScaler()`: Chuẩn hóa dữ liệu về phân phối chuẩn.
- `fit_transform(X_train)`: Áp dụng scaler lên tập train.
- `transform(X_test)`: Dùng cùng scaler để chuẩn hóa tập test.

## 7. Xây dựng mô hình Random Forest

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Giải thích:**

- `RandomForestClassifier(n_estimators=100)`: Sử dụng 100 cây quyết định.
- `fit(X_train, y_train)`: Huấn luyện mô hình trên tập train.

## 8. Dự đoán

```python
y_pred = model.predict(X_test)
```

**Giải thích:**

- `predict(X_test)`: Dự đoán rủi ro tín dụng trên tập test.

## 9. Đánh giá mô hình

```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**Giải thích:**

- `accuracy_score(y_test, y_pred)`: Tính độ chính xác của mô hình.
- `classification_report(y_test, y_pred)`: Hiển thị Precision, Recall, F1-score.
- `confusion_matrix(y_test, y_pred)`: Hiển thị ma trận nhầm lẫn.

---

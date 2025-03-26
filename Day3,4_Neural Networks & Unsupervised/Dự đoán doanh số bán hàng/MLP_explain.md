# MPL Explain

## 1. Giới thiệu

📌 Notebook này sử dụng **Mạng nơ-ron MLP (Multi-Layer Perceptron)** để huấn luyện và đánh giá mô hình dự đoán.

---

## 2. Import thư viện

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

**Giải thích:**

- `numpy` & `pandas`: Hỗ trợ xử lý dữ liệu.
- `matplotlib.pyplot`: Trực quan hóa dữ liệu.
- `tensorflow.keras`: Xây dựng mô hình MLP.
- `Sequential`: Tạo mô hình theo từng lớp.
- `Dense`: Layer fully connected của mạng nơ-ron.
- `train_test_split`: Chia dữ liệu thành tập train/test.
- `StandardScaler`: Chuẩn hóa dữ liệu về khoảng có giá trị trung bình 0 và phương sai 1.

---

## 3. Đọc dữ liệu

```python
df = pd.read_csv("data.csv")
print(df.head())
```

**Giải thích:**

- Đọc dữ liệu từ file `data.csv`.
- Hiển thị 5 dòng đầu tiên để kiểm tra dữ liệu.

---

## 4. Tiền xử lý dữ liệu

```python
df = df.dropna()
X = df.drop(columns=["target"])
y = df["target"]
```

**Giải thích:**

- `dropna()`: Loại bỏ các dòng chứa giá trị khuyết thiếu.
- `X`: Chứa các đặc trưng đầu vào.
- `y`: Nhãn cần dự đoán.

---

## 5. Chia tập dữ liệu

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Giải thích:**

- `test_size=0.2`: 20% dữ liệu dành cho kiểm tra, 80% để huấn luyện.
- `random_state=42`: Đảm bảo kết quả có thể tái lập.

---

## 6. Chuẩn hóa dữ liệu

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Giải thích:**

- `StandardScaler()`: Chuẩn hóa dữ liệu giúp mô hình học hiệu quả hơn.
- `fit_transform(X_train)`: Áp dụng chuẩn hóa trên tập train.
- `transform(X_test)`: Áp dụng chuẩn hóa trên tập test bằng cùng scaler.

---

## 7. Xây dựng mô hình MLP

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Giải thích:**

- `Sequential()`: Khởi tạo mô hình theo từng lớp.
- `Dense(64, activation='relu')`: Lớp ẩn đầu tiên có 64 neuron, dùng ReLU.
- `Dense(32, activation='relu')`: Lớp ẩn thứ hai có 32 neuron.
- `Dense(1, activation='sigmoid')`: Lớp đầu ra, dùng Sigmoid vì bài toán nhị phân.
- `compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`: Cấu hình tối ưu hóa và hàm mất mát.

---

## 8. Huấn luyện mô hình

```python
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
```

**Giải thích:**

- `epochs=20`: Huấn luyện mô hình trong 20 vòng lặp.
- `batch_size=16`: Mỗi lần cập nhật trọng số sử dụng 16 mẫu dữ liệu.
- `validation_data`: Kiểm tra trên tập test.

---

## 9. Đánh giá mô hình

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
```

**Giải thích:**

- `model.evaluate()`: Đánh giá mô hình trên tập kiểm tra.
- `print(f'Test Accuracy: {accuracy}')`: Hiển thị độ chính xác trên tập test.

---

## 10. Vẽ biểu đồ loss và accuracy

```python
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

**Giải thích:**

- `history.history['loss']`: Giá trị loss của tập huấn luyện.
- `history.history['val_loss']`: Giá trị loss của tập kiểm tra.
- `plt.plot()`: Vẽ biểu đồ giúp quan sát quá trình học của mô hình.

---

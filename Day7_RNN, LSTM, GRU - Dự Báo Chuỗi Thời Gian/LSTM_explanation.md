# Explain LSTM

## 1. Import thư viện

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```

**Giải thích:**

- `numpy` và `pandas`: Dùng để xử lý dữ liệu dạng số và bảng.
- `tensorflow` và `keras`: Dùng để xây dựng mô hình học sâu.
- `Sequential`: Kiểu mô hình đơn giản gồm các layer xếp chồng lên nhau.
- `LSTM`: Layer LSTM giúp xử lý chuỗi dữ liệu.
- `Dense`: Layer fully connected để tạo đầu ra của mạng.

## 2. Load và xử lý dữ liệu

```python
df = pd.read_csv("data.csv")
df.head()
```

**Giải thích:**

- Đọc dữ liệu từ file `data.csv`.
- Hiển thị 5 dòng đầu tiên của dataset.

## 3. Tiền xử lý dữ liệu

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
```

**Giải thích:**

- `MinMaxScaler()`: Chuẩn hóa dữ liệu về khoảng [0,1] để LSTM hoạt động hiệu quả hơn.
- `fit_transform(df)`: Chuẩn hóa toàn bộ dữ liệu.

## 4. Chia dữ liệu thành tập train và test

```python
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]
```

**Giải thích:**

- 80% dữ liệu dùng để train, 20% dùng để test.
- `train_data` chứa phần đầu của dữ liệu, `test_data` chứa phần còn lại.

## 5. Chuẩn bị dữ liệu cho mô hình LSTM

```python
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 10  # Số bước thời gian
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
```

**Giải thích:**

- `create_sequences(data, seq_length)`: Hàm này tạo ra các mẫu dữ liệu đầu vào (`X`) và nhãn (`y`).
- **Cách hoạt động:**
  - Duyệt qua toàn bộ dữ liệu với bước nhảy 1.
  - Lấy `seq_length` phần tử liên tiếp làm đầu vào (`X`).
  - Lấy phần tử ngay sau đó làm nhãn dự đoán (`y`).
- Ví dụ nếu `data = [1, 2, 3, 4, 5]` và `seq_length = 3`, thì tập dữ liệu sẽ như sau:
  - `X = [[1,2,3], [2,3,4]]`, `y = [4, 5]`.
- `seq_length = 10`: Mô hình sẽ sử dụng 10 bước thời gian trước đó để dự đoán bước tiếp theo.

## 6. Xây dựng mô hình LSTM

```python
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

**Giải thích:**

- **Layer LSTM đầu tiên**:
  - `LSTM(50, activation='relu', return_sequences=True)`: 50 đơn vị LSTM, kích hoạt ReLU.
  - `return_sequences=True`: Vì chúng ta có thêm một lớp LSTM tiếp theo nên cần trả về toàn bộ chuỗi đầu ra.
- **Layer LSTM thứ hai**:
  - `LSTM(50, activation='relu')`: 50 đơn vị LSTM, nhưng không có `return_sequences=True`, vì đây là lớp LSTM cuối cùng.
- **Layer Dense (Fully Connected)**:
  - `Dense(1)`: Tầng đầu ra, dự đoán một giá trị duy nhất.
- **Biên dịch mô hình**:
  - `optimizer='adam'`: Sử dụng thuật toán tối ưu Adam.
  - `loss='mse'`: Hàm lỗi Mean Squared Error (MSE), phù hợp cho bài toán dự đoán giá trị liên tục.

## 7. Huấn luyện mô hình

```python
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)
```

**Giải thích:**

- `epochs=20`: Mô hình sẽ lặp lại quá trình huấn luyện 20 lần.
- `batch_size=16`: Mỗi lần cập nhật trọng số sử dụng 16 mẫu dữ liệu.
- `validation_data=(X_test, y_test)`: Sử dụng tập kiểm tra để theo dõi độ chính xác của mô hình.
- `verbose=1`: Hiển thị tiến trình huấn luyện.

**Biểu đồ loss theo epochs**:

```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

**Giải thích:**

- `history.history['loss']` chứa giá trị loss của tập huấn luyện qua từng epoch.
- `history.history['val_loss']` chứa giá trị loss của tập kiểm tra.
- `plt.plot(...)`: Vẽ biểu đồ để xem quá trình giảm loss theo thời gian.

## 8. Dự đoán

```python
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
```

**Giải thích:**

- Dự đoán giá trị từ tập test.
- `inverse_transform(y_pred)`: Chuyển dữ liệu về giá trị ban đầu.

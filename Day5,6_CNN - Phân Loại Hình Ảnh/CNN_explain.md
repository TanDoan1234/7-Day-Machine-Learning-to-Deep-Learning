# Giải thích từng phần của mã nguồn trong CNN.ipynb

## 1. Import thư viện

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

**Giải thích:**

- `numpy` & `pandas`: Hỗ trợ xử lý dữ liệu.
- `matplotlib.pyplot`: Trực quan hóa dữ liệu.
- `tensorflow.keras`: Xây dựng mô hình CNN.
- `Conv2D`: Lớp tích chập để trích xuất đặc trưng từ ảnh.
- `MaxPooling2D`: Lớp pooling giảm kích thước dữ liệu.
- `Flatten`: Chuyển ma trận thành vector để đưa vào lớp Dense.
- `Dense`: Lớp fully connected trong mạng nơ-ron.
- `Dropout`: Giảm overfitting.
- `ImageDataGenerator`: Tiền xử lý và tạo dữ liệu ảnh.

---

## 2. Tiền xử lý dữ liệu ảnh

```python
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory("dataset/train", target_size=(64, 64), batch_size=32, class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory("dataset/train", target_size=(64, 64), batch_size=32, class_mode='categorical', subset='validation')
```

**Giải thích:**

- `rescale=1./255`: Chuẩn hóa giá trị pixel về [0,1].
- `validation_split=0.2`: Chia 20% dữ liệu làm tập validation.
- `flow_from_directory()`: Load dữ liệu ảnh từ thư mục.
- `target_size=(64,64)`: Resize ảnh về 64x64 pixels.

---

## 3. Xây dựng mô hình CNN

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

**Giải thích:**

- `Conv2D(32, (3,3), activation='relu')`: Lớp tích chập với 32 filter kích thước 3x3.
- `MaxPooling2D(2,2)`: Lớp pooling giảm kích thước dữ liệu xuống một nửa.
- `Flatten()`: Chuyển ma trận thành vector.
- `Dense(128, activation='relu')`: Lớp fully connected với 128 neuron.
- `Dropout(0.5)`: Loại bỏ 50% neuron ngẫu nhiên để tránh overfitting.
- `Dense(10, activation='softmax')`: Lớp đầu ra 10 lớp, sử dụng softmax cho phân loại nhiều lớp.

---

## 4. Compile và huấn luyện mô hình

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=20, validation_data=val_data)
```

**Giải thích:**

- `optimizer='adam'`: Dùng thuật toán Adam để tối ưu.
- `loss='categorical_crossentropy'`: Hàm mất mát cho phân loại đa lớp.
- `metrics=['accuracy']`: Theo dõi độ chính xác.
- `epochs=20`: Huấn luyện trong 20 vòng.

---

## 5. Đánh giá mô hình

```python
loss, acc = model.evaluate(val_data)
print(f'Validation Accuracy: {acc}')
```

**Giải thích:**

- `model.evaluate()`: Kiểm tra độ chính xác trên tập validation.

---

## 6. Vẽ biểu đồ loss và accuracy

```python
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

**Giải thích:**

- `history.history['loss']`: Lưu giá trị loss của tập train.
- `history.history['val_loss']`: Lưu giá trị loss của tập validation.
- `plt.plot()`: Vẽ biểu đồ để quan sát quá trình học.

---

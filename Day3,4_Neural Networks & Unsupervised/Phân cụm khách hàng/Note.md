# Note

## 1. Import thư viện

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

**Giải thích:**

- `numpy` và `pandas`: Hỗ trợ xử lý dữ liệu.
- `matplotlib.pyplot` và `seaborn`: Trực quan hóa dữ liệu.
- `KMeans` từ `sklearn.cluster`: Thuật toán phân cụm K-Means.

---

## 2. Đọc dữ liệu

```python
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
```

**Giải thích:**

- Đọc dữ liệu từ file CSV.
- Hiển thị 5 dòng đầu tiên để kiểm tra dữ liệu.

---

## 3. Kiểm tra thông tin dữ liệu

```python
print(df.info())
print(df.describe())
```

**Giải thích:**

- `df.info()`: Kiểm tra kiểu dữ liệu và giá trị thiếu.
- `df.describe()`: Thống kê dữ liệu số như trung bình, min, max.

---

## 4. Trực quan hóa phân phối dữ liệu

```python
sns.pairplot(df, hue="Genre")
plt.show()
```

**Giải thích:**

- `pairplot()`: Hiển thị mối quan hệ giữa các biến số.
- `hue="Genre"`: Phân biệt nam/nữ trong dữ liệu.

---

## 5. Chọn số cụm tối ưu với Elbow Method

```python
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df.iloc[:, 2:])
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.xlabel('Số cụm')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**Giải thích:**

- Dùng Elbow Method để tìm số cụm tối ưu.
- `inertia`: Tổng bình phương khoảng cách trong cụm.
- Vẽ biểu đồ để xác định "elbow point".

---

## 6. Áp dụng thuật toán K-Means

```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df.iloc[:, 2:])
```

**Giải thích:**

- `n_clusters=5`: Chọn 5 cụm dựa vào Elbow Method.
- `fit_predict()`: Áp dụng K-Means và thêm nhãn cụm vào dataset.

---

## 7. Trực quan hóa kết quả phân cụm

```python
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['Annual Income'], y=df['Spending Score'], hue=df['Cluster'], palette='Set1')
plt.title("Phân cụm khách hàng")
plt.xlabel("Thu nhập hàng năm")
plt.ylabel("Điểm chi tiêu")
plt.show()
```

**Giải thích:**

- Vẽ biểu đồ phân cụm theo Thu nhập và Điểm chi tiêu.
- Màu sắc đại diện cho từng cụm khách hàng.

---

# Bài tập Data Mining: Decision Tree, Naive Bayes và K-means

Dự án gồm 3 bài toán cơ bản trong Data Mining/Machine Learning:

1. **Decision Tree ID3**
2. **Naive Bayes**
3. **K-means Clustering**

Mục tiêu chính của các chương trình là đọc dữ liệu từ file CSV, thực hiện tính toán theo từng thuật toán, in chi tiết các bước xử lý và đưa ra kết quả cuối cùng.

---

## 1. Cấu trúc thư mục đề xuất

```text
DTMN/
├── decision_tree_root_entropy_ig.py
├── naive_bayes.py
├── kmeans.py
├── data.csv
└── README.md
```

Tùy từng bài toán, file `data.csv` sẽ có cấu trúc khác nhau.

---

# Bài toán 1: Decision Tree ID3

## 1.1. Mục tiêu

Bài toán Decision Tree dùng thuật toán **ID3** để xây dựng cây quyết định dựa trên:

- Entropy
- Weighted Entropy
- Information Gain

Chương trình sẽ tự động tính Entropy, tính Information Gain cho từng thuộc tính, chọn thuộc tính có Information Gain cao nhất làm node và tiếp tục xây cây cho đến khi tạo được cây quyết định hoàn chỉnh.

## 1.2. Dữ liệu đầu vào

File `data.csv` có dạng:

```csv
id,outlook,terrain,temperature,humidity,wind,play
1,sunny,slope,hot,high,weak,no
2,sunny,undulating,hot,low,strong,no
3,overcast,slope,hot,high,low,yes
```

Trong đó:

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã dòng dữ liệu, không dùng để chia cây |
| `outlook`, `terrain`, `temperature`, `humidity`, `wind` | Các thuộc tính đầu vào |
| `play` | Cột nhãn cần dự đoán |

Mặc định:

```python
TARGET = "play"
CSV_FILE = "data.csv"
```

## 1.3. Ý tưởng thuật toán

Thuật toán ID3 hoạt động theo các bước:

```text
1. Tính Entropy của tập dữ liệu hiện tại.
2. Với từng thuộc tính, chia dữ liệu theo các giá trị của thuộc tính đó.
3. Tính Entropy có trọng số sau khi chia.
4. Tính Information Gain.
5. Chọn thuộc tính có Information Gain cao nhất làm node.
6. Lặp lại quá trình cho các nhánh con.
7. Dừng khi tập dữ liệu đã thuần hoặc không còn thuộc tính để chia.
```

Công thức Entropy:

```text
Entropy(S) = -Σ p(c) * log2(p(c))
```

Công thức Weighted Entropy:

```text
Entropy(feature) = Σ (|Sv| / |S|) * Entropy(Sv)
```

Công thức Information Gain:

```text
IG(feature) = Entropy(S) - Entropy(feature)
```

## 1.4. Kết quả đầu ra

Chương trình in ra:

- Entropy ban đầu của tập dữ liệu.
- Entropy và Information Gain của từng thuộc tính.
- Thuộc tính được chọn làm root node.
- Các bước xây dựng cây quyết định.
- Cây quyết định cuối cùng.

## 1.5. Cách chạy

```powershell
python decision_tree_root_entropy_ig.py
```

Hoặc nếu muốn đổi target:

```powershell
python decision_tree_root_entropy_ig.py --target play --ignore id
```

---

# Bài toán 2: Naive Bayes

## 2.1. Mục tiêu

Bài toán Naive Bayes dùng để phân loại mẫu mới dựa trên xác suất.

Với cùng bộ dữ liệu categorical, chương trình sẽ tính:

- Xác suất tiên nghiệm `P(class)`
- Xác suất có điều kiện `P(feature=value | class)`
- Score của từng class
- Nhãn dự đoán cuối cùng

## 2.2. Dữ liệu đầu vào

File `data.csv` có thể dùng cùng dạng với bài Decision Tree:

```csv
id,outlook,terrain,temperature,humidity,wind,play
1,sunny,slope,hot,high,weak,no
2,sunny,undulating,hot,low,strong,no
3,overcast,slope,hot,high,low,yes
```

Trong đó:

| Cột | Ý nghĩa |
|---|---|
| `outlook`, `terrain`, `temperature`, `humidity`, `wind` | Các feature dùng để dự đoán |
| `play` | Nhãn cần dự đoán |
| `id` | Cột bỏ qua |

## 2.3. Mẫu cần dự đoán

Người dùng sửa trực tiếp trong code tại biến:

```python
SAMPLES_TO_PREDICT = [
    {
        "outlook": "sunny",
        "terrain": "flat",
        "temperature": "cool",
        "humidity": "normal",
        "wind": "weak"
    }
]
```

Nếu đề cho nhiều mẫu, thêm nhiều dictionary vào list:

```python
SAMPLES_TO_PREDICT = [
    {
        "outlook": "sunny",
        "terrain": "flat",
        "temperature": "cool",
        "humidity": "normal",
        "wind": "weak"
    },
    {
        "outlook": "rainy",
        "terrain": "slope",
        "temperature": "mild",
        "humidity": "high",
        "wind": "weak"
    }
]
```

## 2.4. Ý tưởng thuật toán

Naive Bayes tính score cho từng class:

```text
Score(class) =
P(class)
* P(x1 | class)
* P(x2 | class)
* ...
* P(xn | class)
```

Trong đó:

- `P(class)` là xác suất tiên nghiệm.
- `P(xi | class)` là xác suất có điều kiện.
- Class nào có score lớn nhất thì được chọn làm kết quả dự đoán.

## 2.5. Laplace smoothing

Chương trình dùng Laplace smoothing để tránh xác suất bằng 0.

Công thức:

```text
P = (count + 1) / (total_class + k)
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| `count` | Số mẫu có `feature=value` và thuộc class đang xét |
| `total_class` | Tổng số mẫu thuộc class đang xét |
| `k` | Số giá trị khác nhau của feature |

## 2.6. Kết quả đầu ra

Chương trình in ra:

- `P(play=yes)`, `P(play=no)`
- Bảng xác suất có điều kiện `P(feature=value | play=yes/no)`
- Chi tiết score của từng mẫu cần dự đoán
- Kết luận nhãn cuối cùng

## 2.7. Cách chạy

```powershell
python naive_bayes.py
```

---

# Bài toán 3: K-means Clustering

## 3.1. Mục tiêu

Bài toán K-means dùng để phân cụm dữ liệu điểm 2 chiều.

Chương trình sẽ:

- Đọc dữ liệu điểm từ file CSV.
- Nhập `k` từ bàn phím.
- Nhập MSSV từ bàn phím.
- Lấy `k` chữ số cuối MSSV làm tâm ban đầu.
- Lặp cho đến khi tâm cụm không đổi.
- In bảng khoảng cách, kết quả gán cụm và tâm mới sau mỗi vòng.
- Vẽ biểu đồ minh họa các cụm.

## 3.2. Dữ liệu đầu vào

File `data.csv` cho K-means cần có dạng:

```csv
STT,x1,x2
0,1,5
1,3,5
2,2,14
3,3,4
4,12,4
5,2,6
6,4,1
7,4,14
8,4,7
9,4,6
10,7,1
11,6,5
12,8,1
13,4,8
14,8,3
15,5,5
16,3,12
17,7,13
```

Trong đó:

| Cột | Ý nghĩa |
|---|---|
| `STT` | Mã điểm dữ liệu |
| `x1` | Tọa độ thứ nhất |
| `x2` | Tọa độ thứ hai |

## 3.3. Tâm ban đầu

Chương trình nhập:

```text
Nhập số cụm k
Nhập MSSV
```

Sau đó lấy `k` chữ số cuối MSSV làm STT tâm ban đầu.

Ví dụ:

```text
k = 3
MSSV = 23133017
```

Lấy 3 chữ số cuối:

```text
017
```

Tâm ban đầu là:

```text
node 0, node 1, node 7
```

Nếu các chữ số bị trùng, chương trình sẽ tự chọn node chưa dùng tiếp theo, ưu tiên từ node 10, 11, 12, ...

Ví dụ:

```text
k = 3
MSSV = 23133111
```

Ba chữ số cuối là:

```text
111
```

Tâm ban đầu sẽ là:

```text
node 1, node 10, node 11
```

## 3.4. Ý tưởng thuật toán

Mỗi vòng lặp K-means gồm 2 bước chính:

```text
1. Gán cụm:
   - Tính khoảng cách từ từng điểm đến các tâm cụm.
   - Điểm được gán vào cụm có khoảng cách nhỏ nhất.
   - Nếu khoảng cách bằng nhau, chọn cụm có số thứ tự nhỏ hơn.

2. Cập nhật tâm:
   - Tâm mới là trung bình tọa độ của các điểm trong cụm.
```

Công thức khoảng cách Euclid:

```text
d(A, B) = sqrt((xA - xB)^2 + (yA - yB)^2)
```

Công thức cập nhật tâm:

```text
Cx = trung bình các x trong cụm
Cy = trung bình các y trong cụm
```

## 3.5. Điều kiện dừng

Chương trình không giới hạn số vòng lặp cố định.

Thuật toán dừng khi:

```text
Tâm cụm mới giống tâm cụm cũ
```

Vì code làm tròn tâm đến 2 chữ số thập phân, nếu sau khi làm tròn mà tâm không đổi thì xem như hội tụ.

## 3.6. Kết quả đầu ra

Chương trình in ra sau mỗi vòng:

- Bảng khoảng cách từ từng điểm đến từng tâm cụm.
- Cụm được gán cho từng điểm.
- Danh sách điểm thuộc từng cụm.
- Tâm cụm mới.
- Biểu đồ minh họa cụm sau mỗi vòng.

Cuối cùng chương trình in:

- Tâm cụm cuối cùng.
- Danh sách điểm thuộc từng cụm.
- Kết luận thuật toán đã hội tụ.

## 3.7. Cách chạy

```powershell
python kmeans.py
```

Ví dụ nhập:

```text
Nhập số cụm k: 3
Nhập MSSV: 23133017
```

---

# Ghi chú chung

## Yêu cầu môi trường

Cần cài Python 3.

Riêng bài K-means có vẽ biểu đồ nên cần cài thêm `matplotlib`:

```powershell
pip install matplotlib
```

## Lưu ý về file CSV

Bài Decision Tree và Naive Bayes dùng file CSV dạng categorical:

```text
id,outlook,terrain,temperature,humidity,wind,play
```

Bài K-means dùng file CSV dạng tọa độ:

```text
STT,x1,x2
```

Vì vậy nếu chạy từng bài, cần đảm bảo file `data.csv` đúng cấu trúc tương ứng với bài đó.

Một cách rõ ràng hơn là tách thành 2 file:

```text
data_bayes_tree.csv
data_kmeans.csv
```

Sau đó sửa biến `CSV_FILE` trong từng file `.py` cho phù hợp.

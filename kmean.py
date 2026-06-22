import numpy as np
import pandas as pd

# DATA
X = np.array([
    [1, 5], [3, 5], [2, 7], [3, 4], [5, 5], [2, 6], [2, 3],
    [4, 6], [7, 1], [6, 5], [8, 1], [4, 8], [8, 3], [7, 5]
])

# 1. KHỞI TẠO TÂM CỤM BAN ĐẦU 
def init_centroids(X, mssv, k):
    
    if k > len(mssv):
        raise ValueError("k không được lớn hơn độ dài MSSV")
    
    # Lấy k ký tự cuối mssv và chuyển thành số
    last_digits = mssv[-k:]

    # Xử lý trùng (ví dụ 111, 222, 333,....)
    stt = []
    seen = set()
    next_node = 10  # node phụ bắt đầu từ 10

    # Xử lý trùng lặp (ví dụ 111, 333,...)
    for ch in last_digits:
        num = int(ch)

        # Nếu chưa xuất hiện thì chạy if
        if num not in seen:
            seen.add(num)
            stt.append(num)
        else:
            # từ lần trùng thứ 2 trở đi → dùng node 10,11,...
            stt.append(next_node)
            next_node += 1

    # Lấy các điểm tương ứng làm tâm cụm ban đầu
    C = np.array([X[i] for i in stt], dtype=float)

    return C, stt


# 2. GÁN MỖI ĐIỂM VÀO CỤM GẦN NHẤT
def assign_clusters(X, C):
    clusters = []     # lưu cụm của từng điểm
    distances = []    # lưu khoảng cách đến từng cụm

    for x in X:

        # Tính khoảng cách từ x đến từng tâm cụm
        dist = [np.sqrt((x[0] - c[0])**2 + (x[1] - c[1])**2) for c in C]
        distances.append(dist)

        # Chọn cụm có khoảng cách nhỏ nhất
        clusters.append(np.argmin(dist) + 1)

    return np.array(clusters), distances


# 3. CẬP NHẬT TÂM CỤM (MEAN OF POINTS)
def update_centroids(X, clusters, C, k):

    new_C = np.zeros((k, 2))  # tạo mảng chứa tâm mới

    for i in range(k):
        points = X[clusters == (i + 1)]  # Lọc tất cả điểm thuộc cụm i+1

        if len(points) > 0:
            new_C[i] = np.mean(points, axis=0)
        else:
            new_C[i] = C[i]

    return np.round(new_C, 2)


# 4. HIỂN THỊ BẢNG (GIỐNG EXCEL)
def show_table(X, clusters, distances, k):
    df = pd.DataFrame(X, columns=['x1', 'x2'])     # tạo bảng dữ liệu
    df.insert(0, 'STT', range(len(X)))             # thêm STT (index của điểm)

    # thêm khoảng cách tới từng cụm
    for i in range(k):
        df[f'D đến C{i+1}'] = [d[i] for d in distances]

    df['Cụm'] = [f'C{c}' for c in clusters]        # thêm cột kết quả cụm
    print(df.to_string(index=False))


# 5. VÒNG LẶP K-MEANS 
def kmeans(X, mssv, k):

    # Khởi tạo tâm cụm ban đầu từ MSSV
    C, stt = init_centroids(X, mssv, k)

    print("\n TÂM CỤM BAN ĐẦU:")
    for i in range(k):
        print(f"C{i+1} = {C[i]} (từ STT {stt[i]})")

    iteration = 1

    # lặp cho tới khi hội tụ
    while True:

        print(f"\n{'='*60}")
        print(f"LẦN LẶP {iteration}")
        print(f"{'='*60}")

        clusters, distances = assign_clusters(X, C) # bước 1: gán cụm

        show_table(X, clusters, distances, k) # hiển thị bảng để dễ kiểm tra

        new_C = update_centroids(X, clusters, C, k) # bước 2: tính tâm mới

        print("\n TÂM CỤM MỚI:") # in ra trạng thái cụm mới
        for i in range(k):

            stt_list = np.where(clusters == (i + 1))[0] # lấy danh sách điểm thuộc cụm
            print(f"C{i+1} = {new_C[i]} | STT: {stt_list.tolist()}")

        if np.array_equal(C, new_C):
            print("\n THUẬT TOÁN ĐÃ HỘI TỤ!")
            break

        # cập nhật tâm cụm để lặp tiếp
        C = new_C
        iteration += 1


# 6. CHẠY CHƯƠNG TRÌNH
mssv = input("Nhập MSSV: ")
k = int(input("Nhập số cụm k: "))

kmeans(X, mssv, k)
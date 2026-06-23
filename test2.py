import numpy as np
import pandas as pd

# DATA
X = np.array([
    [1, 5], [3, 5], [2, 7], [3, 4], [5, 5], [2, 6], [2, 3],
    [4, 6], [7, 1], [6, 5], [8, 1], [4, 8], [8, 3], [7, 5]
])

# =========================
# 1. INIT CENTROID
# =========================
def init_centroids(X, mssv, k):
    if k > len(mssv):
        raise ValueError("k không được lớn hơn độ dài MSSV")

    last_digits = mssv[-k:]

    stt = []
    seen = set()
    next_node = 10

    for ch in last_digits:
        num = int(ch)

        if num not in seen:
            seen.add(num)
            stt.append(num)
        else:
            stt.append(next_node)
            next_node += 1

    C = np.array([X[i] for i in stt], dtype=float)
    return C, stt


# =========================
# 2. ASSIGN CLUSTERS (FIX NHẸ)
# =========================

def assign_clusters(X, C, limits):
    limits = [5, 5, 4]
    clusters = []
    distances = []
    count = [0] * len(C)

    for x in X:
        dist = [np.linalg.norm(x - c) for c in C]
        distances.append(dist)

        # sort cluster theo gần → xa
        sorted_idx = np.argsort(dist)

        chosen = None

        # chọn cluster gần nhất nhưng còn chỗ
        for idx in sorted_idx:
            if count[idx] < limits[idx]:
                chosen = idx
                break

        clusters.append(chosen + 1)
        count[chosen] += 1

    return np.array(clusters), distances


# =========================
# 3. UPDATE CENTROID (FIX SAFE)
# =========================
def update_centroids(X, clusters, C, k):
    new_C = np.zeros((k, 2))

    for i in range(k):
        points = X[clusters == (i + 1)]

        if len(points) > 0:
            new_C[i] = np.mean(points, axis=0)
        else:
            new_C[i] = C[i]

    return np.round(new_C, 2)


# =========================
# 4. SHOW TABLE
# =========================
def show_table(X, clusters, distances, k):
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df.insert(0, 'STT', range(len(X)))

    for i in range(k):
        df[f'D to C{i+1}'] = [d[i] for d in distances]

    df['Cluster'] = clusters
    print(df.to_string(index=False))


# =========================
# 5. KMEANS LOOP (FIX STOP CONDITION)
# =========================
def kmeans(X, mssv, k):
    C, stt = init_centroids(X, mssv, k)

    print("\nTÂM BAN ĐẦU:")
    for i in range(k):
        print(f"C{i+1} = {C[i]} (STT {stt[i]})")

    iteration = 1

    while True:
        print("\n" + "="*50)
        print(f"ITERATION {iteration}")
        print("="*50)

        clusters, distances = assign_clusters(X, C, [5, 5, 4])
        show_table(X, clusters, distances, k)

        new_C = update_centroids(X, clusters, C, k)

        print("\nCENTROID UPDATE:")
        for i in range(k):
            idx = np.where(clusters == (i + 1))[0]
            print(f"C{i+1} = {new_C[i]} | points: {idx.tolist()}")

        # FIX QUAN TRỌNG
        if np.allclose(C, new_C):
            print("\nCONVERGED!")
            break

        C = new_C
        iteration += 1


# =========================
# RUN
# =========================
mssv = input("Nhập MSSV: ")
k = int(input("Nhập k: "))

kmeans(X, mssv, k)
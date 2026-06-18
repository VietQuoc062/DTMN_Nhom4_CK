import csv
import math
import matplotlib.pyplot as plt


# ============================================================
# CẤU HÌNH CHÍNH
# ============================================================

CSV_FILE = "dataKmean.csv"
ROUND_DIGITS = 2


def load_data_from_csv(path):
    """Đọc dữ liệu điểm từ file CSV."""
    data = {}

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            stt = int(row["STT"])
            x1 = float(row["x1"])
            x2 = float(row["x2"])

            data[stt] = (x1, x2)

    return data


def euclidean_distance(point_a, point_b):
    """
    Tính khoảng cách Euclid giữa 2 điểm.

    Đây là công thức khoảng cách chính của K-means.

    Công thức:
        d(A, B) = sqrt((x1 - x2)^2 + (y1 - y2)^2)

    Ý nghĩa:
        Điểm nào gần tâm cụm nào nhất thì được gán vào cụm đó.
    """
    x1, y1 = point_a
    x2, y2 = point_b

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def input_k_and_mssv(data):
    """Nhập k và MSSV từ bàn phím."""
    max_k = len(data)

    k = int(input("Nhập số cụm k: ").strip())
    if k <= 0 or k > max_k:
        raise ValueError(f"k phải nằm trong khoảng từ 1 đến {max_k}.")

    mssv = input("Nhập MSSV: ").strip()
    if not mssv.isdigit():
        raise ValueError("MSSV chỉ được chứa chữ số.")

    if len(mssv) < k:
        raise ValueError("MSSV phải có số chữ số lớn hơn hoặc bằng k.")

    return k, mssv


def resolve_initial_center_indices(mssv, k, data):
    """
    Xác định k node tâm ban đầu từ k chữ số cuối của MSSV.

    Quy tắc:
        - Lấy k chữ số cuối MSSV.
        - Mỗi chữ số tương ứng với STT node ban đầu.
        - Nếu node bị trùng, thay bằng node chưa dùng tiếp theo.
        - Ưu tiên node thay thế từ 10, 11, 12, ...
        - Nếu hết node >= 10 thì lấy node chưa dùng nhỏ nhất.

    Ví dụ:
        k = 3, MSSV = 23133017
        k chữ số cuối là 017
        => tâm ban đầu: node 0, node 1, node 7

        k = 3, MSSV = 23133111
        k chữ số cuối là 111
        => node 1, node 10, node 11
    """
    last_digits = mssv[-k:]

    chosen_indices = []
    used_indices = set()
    all_indices = sorted(data.keys())

    replacement_candidates = [
        index for index in all_indices if index >= 10
    ] + [
        index for index in all_indices if index < 10
    ]

    for digit in last_digits:
        index = int(digit)

        if index in data and index not in used_indices:
            chosen_indices.append(index)
            used_indices.add(index)
        else:
            replacement = None

            for candidate in replacement_candidates:
                if candidate not in used_indices:
                    replacement = candidate
                    break

            if replacement is None:
                raise ValueError("Không còn node nào để thay thế tâm bị trùng.")

            chosen_indices.append(replacement)
            used_indices.add(replacement)

    return chosen_indices


def get_initial_centroids(data, center_indices):
    """Lấy tọa độ tâm ban đầu từ danh sách STT node."""
    centroids = {}

    for cluster_id, point_index in enumerate(center_indices, start=1):
        centroids[cluster_id] = data[point_index]

    return centroids


def assign_clusters(data, centroids):
    """
    Gán mỗi điểm dữ liệu vào cụm gần nhất.

    Đây là bước 1 của mỗi vòng lặp K-means.

    Cách làm:
        1. Với từng điểm, tính khoảng cách đến tất cả tâm cụm.
        2. Chọn cụm có khoảng cách nhỏ nhất.
        3. Nếu khoảng cách bằng nhau, chọn cụm có số thứ tự nhỏ hơn.
    """
    clusters = {
        cluster_id: []
        for cluster_id in centroids
    }

    distances_table = {}

    for point_index, point in data.items():
        distances = {}

        for cluster_id, centroid in centroids.items():
            distances[cluster_id] = euclidean_distance(point, centroid)

        nearest_cluster = min(
            distances,
            key=lambda cluster_id: (distances[cluster_id], cluster_id)
        )

        clusters[nearest_cluster].append(point_index)
        distances_table[point_index] = distances

    return clusters, distances_table


def update_centroids(data, clusters, old_centroids):
    """
    Cập nhật tâm cụm mới.

    Đây là bước 2 của mỗi vòng lặp K-means.

    Công thức:
        Cx = trung bình x của các điểm trong cụm
        Cy = trung bình y của các điểm trong cụm

    Theo đề:
        Làm tròn tâm mới đến 2 chữ số thập phân.

    Nếu cụm rỗng:
        Giữ nguyên tâm cũ để tránh lỗi.
    """
    new_centroids = {}

    for cluster_id, point_indices in clusters.items():
        if not point_indices:
            new_centroids[cluster_id] = old_centroids[cluster_id]
            continue

        total_x = sum(data[index][0] for index in point_indices)
        total_y = sum(data[index][1] for index in point_indices)

        mean_x = total_x / len(point_indices)
        mean_y = total_y / len(point_indices)

        new_centroids[cluster_id] = (
            round(mean_x, ROUND_DIGITS),
            round(mean_y, ROUND_DIGITS)
        )

    return new_centroids


def has_converged(old_centroids, new_centroids):
    """
    Kiểm tra điều kiện hội tụ.

    Thuật toán dừng khi tâm cụm mới giống tâm cụm cũ.
    Vì đề yêu cầu làm tròn 2 chữ số thập phân,
    nên nếu sau khi làm tròn mà tâm không đổi thì xem như hội tụ.
    """
    return old_centroids == new_centroids


def print_centroids(title, centroids):
    """In danh sách tâm cụm."""
    print(title)

    for cluster_id, centroid in centroids.items():
        print(f"  C{cluster_id} = {centroid}")


def print_distance_table(distances_table, k):
    """In bảng khoảng cách từ từng điểm đến các tâm cụm."""
    print("\nBẢNG KHOẢNG CÁCH")
    print("-" * 90)

    header = f"{'STT':<6}"
    for cluster_id in range(1, k + 1):
        header += f"{'d(C' + str(cluster_id) + ')':>12}"
    header += f"{'Cụm':>10}"

    print(header)
    print("-" * 90)

    for point_index, distances in distances_table.items():
        nearest_cluster = min(
            distances,
            key=lambda cluster_id: (distances[cluster_id], cluster_id)
        )

        row = f"{point_index:<6}"

        for cluster_id in range(1, k + 1):
            row += f"{distances[cluster_id]:>12.2f}"

        row += f"{'C' + str(nearest_cluster):>10}"

        print(row)


def print_clusters(clusters):
    """In danh sách điểm thuộc từng cụm."""
    print("\nKẾT QUẢ GÁN CỤM")
    print("-" * 72)

    for cluster_id, point_indices in clusters.items():
        points_text = ", ".join(str(index) for index in point_indices)
        print(f"C{cluster_id}: {points_text}")


def plot_clusters(data, centroids, clusters=None, title="K-means"):
    """Vẽ biểu đồ các điểm dữ liệu và tâm cụm."""
    plt.figure(figsize=(8, 6))

    if clusters is None:
        for point_index, point in data.items():
            x, y = point
            plt.scatter(x, y)
            plt.text(x + 0.1, y + 0.1, str(point_index), fontsize=9)
    else:
        for cluster_id, point_indices in clusters.items():
            xs = [data[index][0] for index in point_indices]
            ys = [data[index][1] for index in point_indices]

            plt.scatter(xs, ys, label=f"Cụm C{cluster_id}")

            for index in point_indices:
                x, y = data[index]
                plt.text(x + 0.1, y + 0.1, str(index), fontsize=9)

    for cluster_id, centroid in centroids.items():
        x, y = centroid
        plt.scatter(x, y, marker="X", s=180, edgecolors="black")
        plt.text(x + 0.15, y + 0.15, f"C{cluster_id}", fontsize=12)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()
    plt.show()


def run_kmeans():
    """
    Chạy toàn bộ thuật toán K-means.

    Quy trình:
        1. Đọc dữ liệu từ file CSV.
        2. Nhập k và MSSV.
        3. Lấy k chữ số cuối MSSV làm tâm ban đầu.
        4. Gán mỗi điểm vào tâm gần nhất.
        5. Tính lại tâm mới.
        6. Nếu tâm mới không đổi so với tâm cũ thì dừng.
        7. Nếu tâm còn đổi thì tiếp tục lặp.
    """
    data = load_data_from_csv(CSV_FILE)

    k, mssv = input_k_and_mssv(data)

    center_indices = resolve_initial_center_indices(mssv, k, data)
    centroids = get_initial_centroids(data, center_indices)

    print("\nK-MEANS CLUSTERING")
    print("=" * 72)
    print("File CSV:", CSV_FILE)
    print("k =", k)
    print("MSSV:", mssv)
    print(f"{k} chữ số cuối MSSV:", mssv[-k:])
    print("STT tâm ban đầu:", center_indices)
    print_centroids("Tâm ban đầu:", centroids)

    plot_clusters(
        data,
        centroids,
        clusters=None,
        title="Trạng thái ban đầu"
    )

    iteration = 1

    while True:
        print("\n" + "=" * 72)
        print(f"VÒNG LẶP {iteration}")
        print("=" * 72)

        clusters, distances_table = assign_clusters(data, centroids)

        print_distance_table(distances_table, k)
        print_clusters(clusters)

        new_centroids = update_centroids(
            data,
            clusters,
            centroids
        )

        print()
        print_centroids("Tâm mới:", new_centroids)

        plot_clusters(
            data,
            new_centroids,
            clusters=clusters,
            title=f"Sau vòng lặp {iteration}"
        )

        if has_converged(centroids, new_centroids):
            print("\nKẾT LUẬN:")
            print(
                f"Sau vòng lặp {iteration}, các tâm cụm không đổi. "
                "Thuật toán đã hội tụ."
            )
            break

        centroids = new_centroids
        iteration += 1

    print("\n" + "=" * 72)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 72)
    print_centroids("Tâm cụm cuối:", centroids)
    print_clusters(clusters)


if __name__ == "__main__":
    run_kmeans()
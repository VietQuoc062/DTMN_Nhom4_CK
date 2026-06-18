import argparse
import csv
import math
from collections import Counter, defaultdict


# Cấu hình chính
TARGET = "play"
FEATURES = []
CSV_FILE = "data.csv"


def round2(value):
    """Định dạng số thực với 2 chữ số thập phân."""
    return f"{value:.2f}"


def load_csv_file(path):
    """Đọc dữ liệu từ file CSV."""
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def infer_features(rows, target, ignored_columns):
    """Lấy danh sách feature, bỏ cột target và các cột không dùng."""
    if not rows:
        raise ValueError("CSV không có dòng dữ liệu nào.")

    columns = list(rows[0].keys())

    if target not in columns:
        raise ValueError(f"Không tìm thấy cột target '{target}' trong CSV.")

    ignored = set(ignored_columns)

    return [
        column
        for column in columns
        if column != target and column not in ignored
    ]


def entropy(rows, target=None):
    """
    Tính Entropy của tập dữ liệu hiện tại.

    Đây là phần tính toán gốc của Decision Tree ID3.
    Entropy dùng để đo mức độ hỗn loạn của tập dữ liệu.

    Công thức:
        Entropy(S) = -Σ p(c) * log2(p(c))

    Trong đó:
        p(c) là tỉ lệ mẫu thuộc nhãn c.

    Ý nghĩa:
        - Entropy = 0: tập dữ liệu thuần, tất cả mẫu cùng một nhãn.
        - Entropy càng lớn: tập dữ liệu càng lẫn nhiều nhãn.
        - ID3 sẽ tìm cách chia dữ liệu sao cho Entropy sau khi chia giảm nhiều nhất.
    """
    target = target or TARGET
    total = len(rows)

    if total == 0:
        return 0.0

    label_counts = Counter(row[target] for row in rows)

    result = 0.0
    for count in label_counts.values():
        probability = count / total

        if probability > 0:
            result -= probability * math.log2(probability)

    return result


def label_summary(rows, target=None):
    """Đếm số mẫu theo từng nhãn."""
    target = target or TARGET
    counts = Counter(row[target] for row in rows)

    return ", ".join(
        f"{label}={counts[label]}"
        for label in sorted(counts)
    )


def print_entropy_formula(rows, title):
    """In công thức và kết quả Entropy của một tập dữ liệu."""
    total = len(rows)
    counts = Counter(row[TARGET] for row in rows)
    parts = []

    for label in sorted(counts):
        parts.append(
            f"-({counts[label]}/{total})log2({counts[label]}/{total})"
        )

    print(title)
    print(f"  Số mẫu: {total} ({label_summary(rows)})")
    print(f"  Công thức: Entropy = {' '.join(parts)}")
    print(f"  Kết quả: Entropy = {round2(entropy(rows))}")


def split_by_feature(rows, feature):
    """Chia dữ liệu thành các nhóm theo giá trị của feature."""
    groups = defaultdict(list)

    for row in rows:
        groups[row[feature]].append(row)

    return dict(sorted(groups.items()))


def weighted_entropy(rows, feature):
    """
    Tính Entropy có trọng số khi chia dữ liệu theo một thuộc tính.

    Đây là bước kiểm tra: nếu chọn feature này làm node,
    thì sau khi tách thành các nhánh, dữ liệu còn hỗn loạn bao nhiêu.

    Công thức:
        Entropy(feature) = Σ (|Sv| / |S|) * Entropy(Sv)

    Trong đó:
        S  là tập dữ liệu hiện tại.
        Sv là tập con có giá trị feature = v.

    Ý nghĩa:
        - Giá trị càng nhỏ thì feature chia dữ liệu càng tốt.
        - Vì sau khi chia, các nhánh con trở nên thuần hơn.
    """
    total = len(rows)
    groups = split_by_feature(rows, feature)

    weighted_total = 0.0

    for subset in groups.values():
        weight = len(subset) / total
        weighted_total += weight * entropy(subset)

    return weighted_total


def information_gain(rows, feature, base_entropy):
    """
    Tính Information Gain của một thuộc tính.

    Đây là tiêu chí chính để ID3 chọn node.

    Công thức:
        IG(feature) = Entropy(S) - Entropy(feature)

    Trong đó:
        Entropy(S) là độ hỗn loạn ban đầu.
        Entropy(feature) là độ hỗn loạn còn lại sau khi chia.

    Ý nghĩa:
        - IG càng cao thì feature càng làm giảm Entropy nhiều.
        - Feature có IG cao nhất sẽ được chọn làm node hiện tại.
    """
    return base_entropy - weighted_entropy(rows, feature)


def majority_label(rows):
    """Chọn nhãn xuất hiện nhiều nhất."""
    counts = Counter(row[TARGET] for row in rows)

    return counts.most_common(1)[0][0]


def best_feature_by_ig(rows, features):
    """
    Chọn thuộc tính tốt nhất để chia cây tại node hiện tại.

    Cách làm:
        1. Tính Entropy của tập dữ liệu hiện tại.
        2. Với từng feature, tính Entropy có trọng số.
        3. Tính Information Gain của từng feature.
        4. Chọn feature có Information Gain lớn nhất.

    Đây chính là bước quyết định node trong thuật toán ID3.

    Output:
        best: feature tốt nhất để chia.
        results: bảng lưu Entropy và IG của từng feature.
    """
    base = entropy(rows)
    results = {}

    for feature in features:
        results[feature] = {
            "entropy": weighted_entropy(rows, feature),
            "ig": information_gain(rows, feature, base),
        }

    best = max(results, key=lambda name: results[name]["ig"])

    return best, results


def build_tree(rows, features):
    """
    Xây dựng cây quyết định bằng thuật toán ID3.

    Đây là hàm đệ quy chính để tạo toàn bộ Decision Tree.

    Luồng xử lý:
        1. Nếu tất cả mẫu cùng nhãn:
            -> Tạo nút lá.
        2. Nếu không còn feature để chia:
            -> Chọn nhãn xuất hiện nhiều nhất.
        3. Nếu còn feature:
            -> Chọn feature có Information Gain cao nhất.
            -> Tạo node bằng feature đó.
            -> Chia dữ liệu theo từng giá trị của feature.
            -> Gọi đệ quy để xây các nhánh con.

    Ý nghĩa:
        Mỗi lần gọi hàm là một lần xây một node hoặc một nhánh của cây.
    """
    labels = [row[TARGET] for row in rows]

    if len(set(labels)) == 1:
        return labels[0]

    if not features:
        return majority_label(rows)

    best_feature, _ = best_feature_by_ig(rows, features)

    tree = {
        best_feature: {}
    }

    remaining_features = [
        feature
        for feature in features
        if feature != best_feature
    ]

    for value, subset in split_by_feature(rows, best_feature).items():
        tree[best_feature][value] = build_tree(
            subset,
            remaining_features
        )

    return tree


def print_tree(tree, indent=""):
    """In cây quyết định cuối cùng."""
    if isinstance(tree, str):
        print(f"{indent}-> {TARGET} = {tree}")
        return

    feature = next(iter(tree))
    print(f"{indent}[{feature}]")

    branches = list(tree[feature].items())

    for index, (value, subtree) in enumerate(branches):
        is_last = index == len(branches) - 1
        connector = "`--" if is_last else "|--"
        child_indent = "   " if is_last else "|  "

        if isinstance(subtree, str):
            print(
                f"{indent}{connector} {feature} = {value}: "
                f"{TARGET} = {subtree}"
            )
        else:
            print(f"{indent}{connector} {feature} = {value}:")
            print_tree(subtree, indent + child_indent)


def print_tree_building_steps(rows, features, level=0):
    """
    In từng bước xây dựng cây quyết định ID3.

    Hàm này giúp theo dõi quá trình chọn node.

    Mỗi node sẽ in:
        - Số mẫu hiện tại.
        - Số lượng từng nhãn.
        - Entropy hiện tại.
        - Information Gain của các feature còn lại.
        - Feature được chọn làm node.
        - Các nhánh con được tạo ra.

    Đây là phần quan trọng nếu cần giải thích quá trình xây cây.
    """
    prefix = "  " * level
    labels = [row[TARGET] for row in rows]

    print(
        f"{prefix}Tập hiện tại: {len(rows)} mẫu "
        f"({label_summary(rows)}), Entropy = {round2(entropy(rows))}"
    )

    if len(set(labels)) == 1:
        print(f"{prefix}-> Tất cả mẫu cùng nhãn '{labels[0]}', tạo nút lá.")
        return

    if not features:
        label = majority_label(rows)
        print(f"{prefix}-> Hết thuộc tính, chọn nhãn đa số: {label}.")
        return

    best_feature, results = best_feature_by_ig(rows, features)

    print(f"{prefix}Tính IG cho các thuộc tính còn lại:")

    for feature in features:
        print(
            f"{prefix}  IG({feature}) = "
            f"{round2(results[feature]['ig'])} "
            f"(Entropy có trọng số = "
            f"{round2(results[feature]['entropy'])})"
        )

    print(f"{prefix}-> Chọn node: {best_feature}")

    remaining_features = [
        feature
        for feature in features
        if feature != best_feature
    ]

    for value, subset in split_by_feature(rows, best_feature).items():
        print(f"{prefix}Nhánh {best_feature} = {value}:")
        print_tree_building_steps(
            subset,
            remaining_features,
            level + 1
        )


def print_feature_calculation(rows, feature, base_entropy):
    """
    In chi tiết quá trình tính Entropy và Information Gain của một feature.

    Đây là phần dùng để trình bày bài tính tay của ID3.

    Nội dung in ra:
        1. Chia dữ liệu theo từng giá trị của feature.
        2. Tính Entropy cho từng nhánh con.
        3. Tính Entropy có trọng số của feature.
        4. Tính Information Gain của feature.

    Ví dụ:
        Nếu feature = outlook,
        dữ liệu sẽ được chia thành sunny, rainy, overcast.
        Sau đó tính Entropy cho từng nhóm này.
    """
    total = len(rows)
    groups = split_by_feature(rows, feature)

    print("\n" + "=" * 72)
    print(f"THUỘC TÍNH: {feature}")
    print("-" * 72)

    weighted_parts = []

    for value, subset in groups.items():
        subset_entropy = entropy(subset)
        weight = len(subset) / total

        print(f"Giá trị '{value}':")
        print(f"  Số mẫu: {len(subset)}/{total} ({label_summary(subset)})")
        print(f"  Entropy({feature}={value}) = {round2(subset_entropy)}")
        print(
            "  Đóng góp Entropy có trọng số = "
            f"({len(subset)}/{total}) * {round2(subset_entropy)} = "
            f"{round2(weight * subset_entropy)}"
        )

        weighted_parts.append(
            f"({len(subset)}/{total})*{round2(subset_entropy)}"
        )

    attr_entropy = weighted_entropy(rows, feature)
    ig = information_gain(rows, feature, base_entropy)

    print("\nTổng Entropy có trọng số của thuộc tính:")
    print(f"  Entropy({feature}) = {' + '.join(weighted_parts)}")
    print(f"  Entropy({feature}) = {round2(attr_entropy)}")

    print("\nInformation Gain:")
    print(f"  IG({feature}) = Entropy(S) - Entropy({feature})")
    print(f"  IG({feature}) = {round2(base_entropy)} - {round2(attr_entropy)}")
    print(f"  IG({feature}) = {round2(ig)}")

    return attr_entropy, ig


def parse_args():
    """Đọc tham số target và ignore."""
    parser = argparse.ArgumentParser(
        description=(
            "Tính Entropy, Information Gain và xây dựng cây quyết định ID3 "
            "từ file data.csv."
        )
    )

    parser.add_argument(
        "--target",
        default="play",
        help="Tên cột mục tiêu/cột nhãn. Mặc định: play.",
    )

    parser.add_argument(
        "--ignore",
        nargs="*",
        default=["id"],
        help="Các cột bỏ qua, ví dụ: --ignore id name. Mặc định: id.",
    )

    return parser.parse_args()


def main():
    """Chạy toàn bộ chương trình."""
    global TARGET, FEATURES

    args = parse_args()

    TARGET = args.target

    rows = load_csv_file(CSV_FILE)

    FEATURES = infer_features(
        rows,
        TARGET,
        args.ignore
    )

    base_entropy = entropy(rows)
    results = {}

    print("TÍNH VÀ VẼ DECISION TREE BẰNG ENTROPY VÀ INFORMATION GAIN")
    print("=" * 72)
    print("File CSV:", CSV_FILE)
    print("Biến mục tiêu:", TARGET)
    print("Các thuộc tính độc lập:", ", ".join(FEATURES))
    print()

    print("BƯỚC 1: TÍNH ENTROPY CƠ SỞ CỦA TẬP DỮ LIỆU")
    print("-" * 72)
    print_entropy_formula(rows, "Tập S ban đầu:")

    print("\nBƯỚC 2: TÍNH ENTROPY VÀ INFORMATION GAIN CHO TỪNG THUỘC TÍNH")

    for feature in FEATURES:
        attr_entropy, ig = print_feature_calculation(
            rows,
            feature,
            base_entropy
        )

        results[feature] = {
            "entropy": attr_entropy,
            "ig": ig
        }

    best_feature = max(
        results,
        key=lambda name: results[name]["ig"]
    )

    print("\n" + "=" * 72)
    print("BẢNG TỔNG HỢP INFORMATION GAIN")
    print("-" * 72)
    print(f"{'Thuộc tính':<15}{'Entropy':>12}{'Information Gain':>22}")
    print("-" * 72)

    for feature in FEATURES:
        print(
            f"{feature:<15}"
            f"{round2(results[feature]['entropy']):>12}"
            f"{round2(results[feature]['ig']):>22}"
        )

    print("\nBƯỚC 3: KẾT LUẬN")
    print("-" * 72)
    print(
        f"Thuộc tính có Information Gain cao nhất là '{best_feature}' "
        f"với IG = {round2(results[best_feature]['ig'])}."
    )
    print(f"Vì vậy, Root Node nên chọn là: {best_feature.upper()}")

    print("\n" + "=" * 72)
    print("BƯỚC 4: XÂY DỰNG TOÀN BỘ CÂY QUYẾT ĐỊNH")
    print("-" * 72)

    print_tree_building_steps(rows, FEATURES)

    final_tree = build_tree(rows, FEATURES)

    print("\n" + "=" * 72)
    print("CÂY QUYẾT ĐỊNH CUỐI CÙNG")
    print("-" * 72)

    print_tree(final_tree)


if __name__ == "__main__":
    main()
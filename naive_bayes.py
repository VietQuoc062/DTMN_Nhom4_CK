import argparse
import csv
from collections import Counter

TARGET = "play"
CSV_FILE = "data.csv"

# Các feature đang có trong file data.csv:
# 1. outlook: sunny, overcast, rainy
# 2. terrain: slope, undulating, flat
# 3. temperature: hot, mild, cool
# 4. humidity: high, low, normal
# 5. wind: weak, strong, low

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


FEATURES = []


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


def get_labels(rows):
    """Lấy danh sách các nhãn khác nhau."""
    return sorted(set(row[TARGET] for row in rows))


def get_feature_values(rows, feature):
    """Lấy các giá trị khác nhau của một feature."""
    return sorted(set(row[feature] for row in rows))


def prior_probability(rows):
    """
    Tính xác suất tiên nghiệm P(class).

    Đây là bước đầu tiên của Naive Bayes.
    Xác suất tiên nghiệm cho biết tỉ lệ xuất hiện ban đầu của mỗi nhãn,
    trước khi xét các thuộc tính đầu vào.

    Công thức:
        P(class) = số mẫu thuộc class / tổng số mẫu

    Ví dụ:
        P(play=yes) = số dòng play=yes / tổng số dòng
        P(play=no)  = số dòng play=no  / tổng số dòng
    """
    total = len(rows)
    label_counts = Counter(row[TARGET] for row in rows)

    priors = {}

    for label, count in label_counts.items():
        priors[label] = count / total

    return priors


def conditional_probability(rows, feature, value, label):
    """
    Tính xác suất có điều kiện P(feature=value | class=label).

    Đây là phần tính toán trọng tâm của Naive Bayes.

    Công thức thường:
        P(feature=value | class=label)
        = số mẫu có feature=value và class=label / số mẫu thuộc class=label

    Chương trình tính trực tiếp theo tần suất xuất hiện, không dùng
    Laplace smoothing:
        P = count / total_class

    Trong đó:
        count:
            số mẫu có feature=value và class=label

        total_class:
            tổng số mẫu thuộc class=label

    Nếu count bằng 0 thì xác suất bằng 0. Khi đó score của class cũng bằng 0,
    đúng với cách tính tay không làm trơn xác suất.
    """
    label_rows = [
        row
        for row in rows
        if row[TARGET] == label
    ]

    total_class = len(label_rows)

    count = sum(
        1
        for row in label_rows
        if row[feature] == value
    )

    if total_class == 0:
        raise ValueError(f"Không có mẫu nào thuộc lớp '{label}'.")

    return count / total_class


def train_naive_bayes(rows, features):
    """
    Huấn luyện mô hình Naive Bayes.

    Với dữ liệu categorical, huấn luyện chủ yếu là thống kê xác suất
    từ dữ liệu, không phải tối ưu trọng số như hồi quy.

    Mô hình lưu:
        1. Danh sách nhãn.
        2. Danh sách feature.
        3. Xác suất tiên nghiệm P(class).

    Các xác suất P(feature=value | class) sẽ được tính khi dự đoán.
    """
    model = {
        "labels": get_labels(rows),
        "features": features,
        "priors": prior_probability(rows)
    }

    return model


def predict_naive_bayes(rows, model, sample):
    """
    Dự đoán nhãn cho một mẫu mới bằng Naive Bayes.

    Công thức:
        score(class)
        = P(class)
          * P(x1 | class)
          * P(x2 | class)
          * ...
          * P(xn | class)

    Trong đó:
        x1, x2, ..., xn là các giá trị feature của mẫu cần dự đoán.

    Giả định Naive Bayes:
        Các feature độc lập có điều kiện theo class.
        Vì vậy có thể nhân các xác suất điều kiện lại với nhau.

    Kết luận:
        Class nào có score lớn nhất thì được chọn làm nhãn dự đoán.
    """
    scores = {}

    for label in model["labels"]:
        score = model["priors"][label]

        for feature in model["features"]:
            value = sample[feature]

            prob = conditional_probability(
                rows,
                feature,
                value,
                label
            )

            score *= prob

        scores[label] = score

    predicted_label = max(scores, key=scores.get)

    return predicted_label, scores


def print_prior_probabilities(rows):
    """In xác suất tiên nghiệm P(class)."""
    priors = prior_probability(rows)

    print("BƯỚC 1: TÍNH XÁC SUẤT TIÊN NGHIỆM")
    print("-" * 72)

    for label, prob in priors.items():
        count = sum(1 for row in rows if row[TARGET] == label)

        print(
            f"P({TARGET}={label}) = "
            f"{count}/{len(rows)} = {prob:.4f}"
        )


def print_conditional_probabilities(rows, features):
    """In bảng xác suất có điều kiện P(feature=value | class)."""
    labels = get_labels(rows)

    print("\nBƯỚC 2: TÍNH XÁC SUẤT CÓ ĐIỀU KIỆN")
    print("-" * 72)

    for feature in features:
        values = get_feature_values(rows, feature)

        print(f"\nTHUỘC TÍNH: {feature}")

        for value in values:
            for label in labels:
                prob = conditional_probability(
                    rows,
                    feature,
                    value,
                    label
                )

                count = sum(
                    1
                    for row in rows
                    if row[TARGET] == label and row[feature] == value
                )
                total_class = sum(
                    1 for row in rows if row[TARGET] == label
                )

                print(
                    f"  P({feature}={value} | {TARGET}={label}) = "
                    f"{count}/{total_class} = {prob:.4f}"
                )


def print_prediction_detail(rows, model, sample, sample_index=None):
    """
    In chi tiết quá trình dự đoán một mẫu.

    Với mỗi class, chương trình in:
        1. P(class)
        2. P(feature=value | class) của từng feature
        3. Công thức nhân xác suất
        4. Score cuối cùng

    Sau đó so sánh score các class và chọn class lớn nhất.
    """
    if sample_index is not None:
        print("\n" + "=" * 72)
        print(f"DỰ ĐOÁN MẪU {sample_index}")
        print("=" * 72)
    else:
        print("\n" + "=" * 72)
        print("DỰ ĐOÁN MẪU")
        print("=" * 72)

    print("Mẫu cần dự đoán:")
    for feature, value in sample.items():
        print(f"  {feature} = {value}")

    print()

    scores = {}

    for label in model["labels"]:
        score = model["priors"][label]
        prior_count = sum(1 for row in rows if row[TARGET] == label)

        print(f"Tính score cho {TARGET}={label}:")
        print(
            f"  P({TARGET}={label}) = "
            f"{prior_count}/{len(rows)} = {model['priors'][label]:.4f}"
        )

        formula_parts = [f"({prior_count}/{len(rows)})"]

        for feature in model["features"]:
            value = sample[feature]

            prob = conditional_probability(
                rows,
                feature,
                value,
                label
            )

            score *= prob

            count = sum(
                1
                for row in rows
                if row[TARGET] == label and row[feature] == value
            )
            total_class = sum(
                1 for row in rows if row[TARGET] == label
            )
            formula_parts.append(f"({count}/{total_class})")

            print(
                f"  P({feature}={value} | {TARGET}={label}) = "
                f"{count}/{total_class} = {prob:.4f}"
            )

        print(f"  Score({label}) = {' * '.join(formula_parts)}")
        print(f"  Score({label}) = {score:.8f}")
        print()

        scores[label] = score

    predicted_label = max(scores, key=scores.get)

    print("SO SÁNH SCORE:")
    for label, score in scores.items():
        print(f"  Score({label}) = {score:.8f}")

    print(f"\n=> Dự đoán cuối cùng: {TARGET} = {predicted_label}")


def print_all_predictions(rows, model, samples):
    """In kết quả dự đoán cho nhiều mẫu."""
    print("\nBƯỚC 3: DỰ ĐOÁN MẪU MỚI")
    print("-" * 72)

    for index, sample in enumerate(samples, start=1):
        print_prediction_detail(
            rows,
            model,
            sample,
            sample_index=index
        )


def validate_samples(samples, features):
    """Kiểm tra các mẫu dự đoán có đủ feature hay không."""
    for index, sample in enumerate(samples, start=1):
        missing_features = [
            feature
            for feature in features
            if feature not in sample
        ]

        if missing_features:
            raise ValueError(
                f"Mẫu {index} thiếu feature: {', '.join(missing_features)}"
            )


def parse_args():
    """Đọc tham số target và ignore."""
    parser = argparse.ArgumentParser(
        description="Naive Bayes cho dữ liệu categorical từ file data.csv."
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
        help="Các cột bỏ qua. Mặc định: id.",
    )

    return parser.parse_args()


def main():
    """Chạy toàn bộ chương trình Naive Bayes."""
    global TARGET, FEATURES

    args = parse_args()

    TARGET = args.target

    rows = load_csv_file(CSV_FILE)

    FEATURES = infer_features(
        rows,
        TARGET,
        args.ignore
    )

    validate_samples(SAMPLES_TO_PREDICT, FEATURES)

    model = train_naive_bayes(rows, FEATURES)

    print("NAIVE BAYES CHO DỮ LIỆU CATEGORICAL")
    print("=" * 72)
    print("File CSV:", CSV_FILE)
    print("Biến mục tiêu:", TARGET)
    print("Các thuộc tính độc lập:", ", ".join(FEATURES))
    print()

    print_prior_probabilities(rows)

    print_conditional_probabilities(rows, FEATURES)

    print_all_predictions(rows, model, SAMPLES_TO_PREDICT)


if __name__ == "__main__":
    main()

import pandas as pd

# =========================
# 1. Tạo dữ liệu ban đầu
# =========================
data = [
    [1,  "sunny",    "hot",  "high",   "weak",   "no"],
    [2,  "sunny",    "hot",  "high",   "strong", "no"],
    [3,  "overcast", "hot",  "high",   "weak",   "yes"],
    [4,  "rainy",    "mild", "high",   "weak",   "yes"],
    [5,  "rainy",    "cool", "normal", "weak",   "yes"],
    [6,  "rainy",    "cool", "normal", "strong", "no"],
    [7,  "overcast", "cool", "normal", "strong", "yes"],
    [8,  "sunny",    "mild", "high",   "weak",   "no"],
    [9,  "sunny",    "cool", "normal", "weak",   "yes"],
    [10, "rainy",    "mild", "normal", "weak",   "yes"],
    [11, "sunny",    "mild", "normal", "strong", "yes"],
    [12, "overcast", "mild", "high",   "strong", "yes"],
    [13, "overcast", "hot",  "normal", "weak",   "yes"],
    [14, "rainy",    "mild", "high",   "strong", "no"],
    [15, "sunny",    "cool", "normal", "strong", "no"],
]

columns = ["id", "outlook", "temperature", "humidity", "wind", "play"]
df = pd.DataFrame(data, columns=columns)

# =========================
# 2. Các dòng cần dự đoán
# =========================
test_rows = [
    ("X1", "sunny",    "hot",  "normal", "weak"),
    ("X2", "rainy",    "hot",  "normal", "weak"),
    ("X3", "rainy",    "hot",  "high",   "strong"),
    ("X4", "overcast", "hot",  "high",   "strong"),
    ("X5", "overcast", "cool", "high",   "weak"),
    ("X6", "sunny",    "cool", "high",   "weak"),
]

features = ["outlook", "temperature", "humidity", "wind"]

# =========================
# 3. Các hàm in kết quả
# =========================
def print_class_header(c, class_count, total, prior):
    print(f"\nClass play = '{c}'")
    print(f"P(play='{c}') = {class_count}/{total} = {prior:.6f}")

def print_class_probability(c, prob):
    print(f"=> P(X | play='{c}') * P(play='{c}') = {prob:.6f}")

def print_compare_result(result, predicted_class):
    print("\nKết quả so sánh:")

    for c, p in result.items():
        print(f"play='{c}': {p:.6f}")

    print(f"=> Dự đoán: play = '{predicted_class}'")


def print_added_row(name, predicted):
    print(f"\nĐã thêm {name} vào DataFrame với play = '{predicted}'")

# =========================
# 4. Hàm tính Naive Bayes
# =========================
def predict_naive_bayes(df, x):
    total = len(df)
    classes = df["play"].unique()

    result = {}

    print("\n" + "=" * 60)
    print(f"Dự đoán cho: {x}")
    print("=" * 60)

    for c in classes:
        df_c = df[df["play"] == c]
        class_count = len(df_c)

        # P(play = c)
        prior = class_count / total
        prob = prior

        print_class_header(c, class_count, total, prior)

        # Tính P(xi | play = c)
        for feature, value in zip(features, x):
            count_value = len(df_c[df_c[feature] == value])
            cond_prob = count_value / class_count

            prob = prob * cond_prob

            print(
                f"P({feature}='{value}' | play='{c}') "
                f"= {count_value}/{class_count} = {cond_prob:.6f}"
            )

        print_class_probability(c, prob)

        result[c] = prob

    predicted_class = max(result, key=result.get)

    print_compare_result(result, predicted_class)

    return predicted_class

# =========================
# 5. Dự đoán từng dòng
#    Sau khi dự đoán thì thêm vào DataFrame
# =========================
for name, outlook, temperature, humidity, wind in test_rows:
    x = (outlook, temperature, humidity, wind)

    predicted = predict_naive_bayes(df, x)

    new_row = {
        "id": name,
        "outlook": outlook,
        "temperature": temperature,
        "humidity": humidity,
        "wind": wind,
        "play": predicted
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    print_added_row(name, predicted)

# =========================
# 6. In DataFrame cuối cùng
# =========================
print("\n\nDataFrame cuối cùng:")
print(df)
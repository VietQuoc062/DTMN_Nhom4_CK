import pandas as pd
# =========================
# 1. Tạo DataFrame ban đầu
# =========================
def create_dataframe():
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

    return df

# =========================
# 2. Tạo dữ liệu cần dự đoán
# =========================
def create_test_rows():
    test_rows = [
        {
            "id": "X1",
            "outlook": "sunny",
            "temperature": "hot",
            "humidity": "normal",
            "wind": "weak"
        },
        {
            "id": "X2",
            "outlook": "rainy",
            "temperature": "hot",
            "humidity": "normal",
            "wind": "weak"
        },
        {
            "id": "X3",
            "outlook": "rainy",
            "temperature": "hot",
            "humidity": "high",
            "wind": "strong"
        },
        {
            "id": "X4",
            "outlook": "overcast",
            "temperature": "hot",
            "humidity": "high",
            "wind": "strong"
        },
        {
            "id": "X5",
            "outlook": "overcast",
            "temperature": "cool",
            "humidity": "high",
            "wind": "weak"
        },
        {
            "id": "X6",
            "outlook": "sunny",
            "temperature": "cool",
            "humidity": "high",
            "wind": "weak"
        },
    ]

    return test_rows

# =========================
# 3. Lấy các cột thuộc tính
# =========================
def get_feature_cols(df, id_col, target_col):
    feature_cols = []

    for col in df.columns:
        if col != id_col and col != target_col:
            feature_cols.append(col)

    return feature_cols

# =========================
# 4. Các hàm in kết quả
# =========================
def print_config(feature_cols, target_col):
    print("Các cột thuộc tính dùng để dự đoán:", feature_cols)
    print("Cột kết quả cần dự đoán:", target_col)

def print_predict_title(x):
    print("\n" + "=" * 60)
    print(f"Dự đoán cho: {x}")
    print("=" * 60)

def print_class_header(target_col, c, class_count, total, prior):
    print(f"\nClass {target_col} = '{c}'")
    print(f"P({target_col}='{c}') = {class_count}/{total} = {prior:.6f}")

def print_conditional_probability(feature, value, target_col, c, count_value, class_count, cond_prob):
    print(
        f"P({feature}='{value}' | {target_col}='{c}') "
        f"= {count_value}/{class_count} = {cond_prob:.6f}"
    )

def print_class_probability(target_col, c, prob):
    print(f"=> P(X | {target_col}='{c}') * P({target_col}='{c}') = {prob:.6f}")


def print_compare_result(target_col, result, predicted_class):
    print("\nKết quả so sánh:")

    for c, p in result.items():
        print(f"{target_col}='{c}': {p:.6f}")

    print(f"=> Dự đoán: {target_col} = '{predicted_class}'")

def print_added_row(name, target_col, predicted):
    print(f"\nĐã thêm {name} vào DataFrame với {target_col} = '{predicted}'")


def print_final_dataframe(df):
    print("\n\nDataFrame cuối cùng:")
    print(df)

# =========================
# 5. Hàm tính Naive Bayes
# =========================
def predict_naive_bayes(df, x, feature_cols, target_col):
    total = len(df)
    classes = df[target_col].unique()

    result = {}

    print_predict_title(x)

    for c in classes:
        df_c = df[df[target_col] == c]
        class_count = len(df_c)

        # P(target = c)
        prior = class_count / total
        prob = prior

        print_class_header(target_col, c, class_count, total, prior)

        # Tính P(xi | target = c)
        for feature in feature_cols:
            value = x[feature]

            count_value = len(df_c[df_c[feature] == value])
            cond_prob = count_value / class_count

            prob = prob * cond_prob

            print_conditional_probability(
                feature,
                value,
                target_col,
                c,
                count_value,
                class_count,
                cond_prob
            )

        print_class_probability(target_col, c, prob)

        result[c] = prob

    predicted_class = max(result, key=result.get)

    print_compare_result(target_col, result, predicted_class)

    return predicted_class

# =========================
# 6. Hàm dự đoán tất cả dòng
# =========================
def predict_all_rows(df, test_rows, feature_cols, target_col, id_col):
    for test_row in test_rows:
        predicted = predict_naive_bayes(
            df,
            test_row,
            feature_cols,
            target_col
        )

        new_row = {}

        # Thêm id
        new_row[id_col] = test_row[id_col]

        # Thêm các cột thuộc tính
        for feature in feature_cols:
            new_row[feature] = test_row[feature]

        # Thêm cột kết quả dự đoán
        new_row[target_col] = predicted

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        print_added_row(test_row[id_col], target_col, predicted)

    return df

# =========================
# 7. Hàm main
# =========================
def main():
    df = create_dataframe()
    test_rows = create_test_rows()

    id_col = "id"
    target_col = "play"

    feature_cols = get_feature_cols(df, id_col, target_col)

    print_config(feature_cols, target_col)

    df = predict_all_rows(
        df,
        test_rows,
        feature_cols,
        target_col,
        id_col
    )

    print_final_dataframe(df)

# =========================
# 8. Gọi hàm main
# =========================
if __name__ == "__main__":
    main()
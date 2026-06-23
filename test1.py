import pandas as pd
import math

# 1. TẠO DỮ LIỆU TRỰC TIẾP BẰNG PANDAS DATAFRAME
data = {
    'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy', 'sunny'],
    'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild', 'cool'],
    'Humidity': ['high', 'low', 'high', 'high', 'normal', 'low', 'normal', 'high', 'normal', 'low', 'normal', 'high', 'normal', 'high', 'normal'],
    'Wind': ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong', 'strong'],
    'Play': ['no', 'no', 'yes', 'hesitate', 'yes', 'no', 'yes', 'no', 'yes', 'hesitate', 'yes', 'yes', 'hesitate', 'no', 'no']
}

df = pd.DataFrame(data)

# 2. HÀM TÍNH ENTROPY
def calculate_entropy(target_col):
    """Tính Entropy của một tập dữ liệu dựa trên cột nhãn"""
    counts = target_col.value_counts()
    total = len(target_col)
    entropy = 0
    for count in counts:
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# 3. HÀM TÍNH INFORMATION GAIN CHO 1 THUỘC TÍNH
def calculate_info_gain(data, feature_name, target_name):
    """Tính Info(D_A) và Gain(A)"""
    total_entropy = calculate_entropy(data[target_name])
    
    # Tính Info_A(D)
    values = data[feature_name].value_counts(normalize=False)
    info_A = 0
    for value, count in values.items():
        subset = data[data[feature_name] == value]
        subset_entropy = calculate_entropy(subset[target_name])
        info_A += (count / len(data)) * subset_entropy
        
    gain_A = total_entropy - info_A
    return info_A, gain_A

# 4. HÀM ĐỆ QUY XÂY DỰNG CÂY VÀ IN TRACE LOG
def build_tree_and_trace(data, features, target_name, depth=0):
    print(f"\n--- BƯỚC PHÂN NHÁNH (Độ sâu: {depth}) ---")
    print(f"Số lượng mẫu hiện tại: {len(data)}")
    
    # Đếm số lượng các nhãn
    label_counts = data[target_name].value_counts().to_dict()
    print(f"Phân phối nhãn: {label_counts}")
    
    # Tính Entropy tổng
    total_entropy = calculate_entropy(data[target_name])
    print(f"Entropy(D) = {total_entropy:.4f}")
    
    # Điều kiện dừng 1: Tập dữ liệu đã thuần nhất (Entropy = 0)
    if total_entropy == 0:
        leaf_label = data[target_name].iloc[0]
        print(f"=> TẬP THUẦN NHẤT. Tạo Nút lá: {leaf_label}")
        return leaf_label
    
    # Điều kiện dừng 2: Hết thuộc tính để xét
    if not features:
        # Trả về nhãn xuất hiện nhiều nhất (Majority Voting)
        leaf_label = data[target_name].mode()[0]
        print(f"=> HẾT THUỘC TÍNH. Tạo Nút lá: {leaf_label}")
        return leaf_label

    # Tính Gain cho từng thuộc tính
    print(f"Bảng tính Information Gain:")
    best_feature = None
    max_gain = -1
    ig_list = []
    for feature in features:
        info_A, gain_A = calculate_info_gain(data, feature, target_name)
        print(f"* Thuộc tính '{feature:<12}': Info = {info_A:.4f}, Gain = {gain_A:.4f}")
        ig_list.append((feature,gain_A))

    ig_list.sort(key=lambda x: x[1])
    mid_index = len(ig_list) // 2
    best_feature = ig_list[mid_index][0]
    max_gain = ig_list[mid_index][1]
            
    print(f"=> CHỌN THUỘC TÍNH GỐC: '{best_feature}' (Gain trung vị = {max_gain:.4f})")
    
    # Tạo node
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]
    
    # Tách dữ liệu và đệ quy
    for value in data[best_feature].unique():
        print(f"\n>>> Đi theo nhánh: {best_feature} = {value}")
        subset = data[data[best_feature] == value]
        subtree = build_tree_and_trace(subset, remaining_features, target_name, depth + 1)
        tree[best_feature][value] = subtree
        
    return tree

# 5. HÀM IN CÂY
def print_ascii_tree(tree, indent="", target_name="Play"):
    """
    Hàm in cây quyết định dưới dạng ASCII.
    """
    if not isinstance(tree, dict):
        return
    
    # Lấy tên thuộc tính đang được dùng để phân nhánh (nút hiện tại)
    feature = list(tree.keys())[0]
    print(f"{indent}[{feature}]")
    
    branches = tree[feature]
    keys = list(branches.keys())
    
    for i, val in enumerate(keys):
        # Kiểm tra xem đây có phải là nhánh cuối cùng không để đổi ký tự vẽ cây
        is_last = (i == len(keys) - 1)
        connector = "`-- " if is_last else "|-- "
        
        subtree = branches[val]
        
        if isinstance(subtree, dict):
            # Nếu nhánh con lại là một cây (dict), in tên nhánh và gọi đệ quy tiếp
            print(f"{indent}{connector}{feature} = {val}:")
            # Nếu là nhánh cuối thì phần nối bên dưới thụt bằng khoảng trắng, ngược lại dùng thanh dọc
            extension = "    " if is_last else "|   "
            print_ascii_tree(subtree, indent + extension, target_name)
        else:
            # Nếu nhánh con là nút lá (chuỗi kết quả), in thẳng kết quả ra
            print(f"{indent}{connector}{feature} = {val}: {target_name} = {subtree}")


if __name__ == "__main__":
    print("MÔ PHỎNG TÍNH TOÁN CÂY QUYẾT ĐỊNH (DECISION TREE)")
    
    # Sửa chữ 'play' thành 'Play' cho khớp với tên cột trong Dictionary
    target_col = 'Play' 
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Gọi hàm xây dựng cây
    decision_tree = build_tree_and_trace(df, feature_cols, target_col)
    
    print("CẤU TRÚC CÂY QUYẾT ĐỊNH CUỐI CÙNG:")
    # Sử dụng hàm in ASCII
    print_ascii_tree(decision_tree, target_name=target_col)
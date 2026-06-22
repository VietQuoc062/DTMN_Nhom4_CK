import pandas as pd

# Tạo một chuỗi dữ liệu (Series)
colors = pd.Series(['Đỏ', 'Xanh', 'Đỏ', 'Đỏ', 'Đen', 'Xanh'])

# Đếm tần suất các màu
print(type(colors.value_counts()))
print(colors.value_counts())

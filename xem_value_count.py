import pandas as pd

# Tạo một chuỗi dữ liệu (Series)
data = {
    'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy', 'sunny'],
    'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild', 'cool'],
    'Humidity': ['high', 'low', 'high', 'high', 'normal', 'low', 'normal', 'high', 'normal', 'low', 'normal', 'high', 'normal', 'high', 'normal'],
    'Wind': ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong', 'strong'],
    'Play': ['no', 'no', 'yes', 'hesitate', 'yes', 'no', 'yes', 'no', 'yes', 'hesitate', 'yes', 'yes', 'hesitate', 'no', 'no']
}

df = pd.DataFrame(data)

# Đếm tần suất các màu
print(type(df['Outlook'].value_counts()))
print(df['Outlook'].value_counts())


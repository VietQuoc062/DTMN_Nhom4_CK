

# Chọn cụm có khoảng cách nhỏ nhất
clusters.append(np.argmin(dist) + 1)


# Nếu trường hợp có 2 hoặc 3 cái bằng nhau:
#1. Chọn ngẫu nhiên
import random

min_val = min(dist)
candidates = [i for i, d in enumerate(dist) if d == min_val]

clusters.append(random.choice(candidates) + 1)


#2. Muốn chọn cái thứ 2
min_val = min(dist)
candidates = [i for i, d in enumerate(dist) if d == min_val]

if len(candidates) >= 3:
    clusters.append(candidates[2] + 1)   # lấy cái thứ 3
elif len(candidates) >= 2:
    clusters.append(candidates[1] + 1)   # lấy cái thứ 2
else:
    clusters.append(candidates[0] + 1)   # chỉ có 1 cái


#3. Muốn chọn cái thứ 3
min_val = min(dist)
candidates = [i for i, d in enumerate(dist) if d == min_val]

if len(candidates) > 1:
    clusters.append(candidates[1] + 1)
else:
    clusters.append(candidates[0] + 1)
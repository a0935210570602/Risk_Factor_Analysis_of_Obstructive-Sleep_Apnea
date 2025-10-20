# 資料集
data = [90, 10, 45, 70, 13, 27, 11, 70, 14, 15, 
        13, 75, 50, 30, 80, 40, 29, 13, 9, 7, 
        20, 85, 55, 94]

# 步驟 1: 算平均數
n = len(data)
mean = sum(data) / n

# 步驟 2: 算每個數據與平均數的差平方
squared_diffs = [(x - mean) ** 2 for x in data]

# 步驟 3: 算總和
sum_squared_diffs = sum(squared_diffs)

# 步驟 4: 樣本標準差 (分母 n-1)
sample_variance = sum_squared_diffs / (n - 1)
sample_sd = sample_variance ** 0.5

print("Mean:", mean)
print("Sample Standard Deviation:", sample_sd)
print("Range:", sample_sd-2*mean , " ~", sample_sd+2*mean)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from praatio import textgrid
import os

def get_word_level_data(tg_dir):
    word_data = []
    for file in os.listdir(tg_dir):
        if not file.endswith('.TextGrid'): continue
        tg = textgrid.openTextgrid(os.path.join(tg_dir, file), includeEmptyIntervals=False)
        
        # 获取words层 
        word_tier = tg.getTier('words')
        for entry in word_tier.entries:
            word = entry.label.lower()
            # 过滤掉静音、噪声和特殊符号
            if word in ['', 'sp', 'sil', 'spn', '<unk>'] or not word.isalpha():
                continue
            
            duration = entry.end - entry.start
            if 0.05 < duration < 2.0 and len(word)<15: # 过滤掉异常短或异常长的词
                word_data.append({
                    'word': word,
                    'char_count': len(word), # 自变量 X：字符数
                    'duration': duration     # 因变量 Y：时长
                })
    return pd.DataFrame(word_data)

# 提取数据
df_words = get_word_level_data(f'mfa_output/fr/textgrid')

grouped_means = df_words.groupby('char_count')['duration'].mean().reset_index()

# 过滤掉样本量太少的组
counts = df_words['char_count'].value_counts()
valid_lengths = counts[counts > 5].index
grouped_means = grouped_means[grouped_means['char_count'].isin(valid_lengths)]

x_means = grouped_means['char_count']
y_means = grouped_means['duration']

# 对均值点进行线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(x_means, y_means)

plt.figure(figsize=(9, 6))
# 绘制均值点
plt.scatter(x_means, y_means, color='blue', s=100, marker='o', 
            label='Mean Duration per Word Length', zorder=3)

# 绘制回归线
plt.plot(x_means, intercept + slope * x_means, color='red', 
         linewidth=2, label=f'Linear Trend (R²={r_value**2:.4f})')

# 装饰图表
plt.xlabel('Word Length (Number of Characters)')
plt.ylabel('Average Duration (s)')
plt.title(f'Deterministic Relationship: Word Length vs. Mean Duration (FR)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 在图上标注回归方程
plt.text(x_means.min(), y_means.max()*0.9, 
         f'Equation: $Y = {intercept:.3f} + {slope:.3f}X$', 
         fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

plt.savefig(f'word_mean_regression.jpg')

print(f"Grouped R-squared: {r_value**2:.4f}")
print(f"Slope (Seconds per char): {slope:.4f}")
import os
import pandas as pd
from praatio import textgrid 

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
VOWEL_SYMBOLS = {
    # 基础元音
    'a', 'e', 'i', 'o', 'u', 'y', 'ø', 'œ', 'ɒ', 'ɔ', 'ɛ', 'ɪ', 'ʊ', 'ʌ', 'æ', 'ɑ', 'ə', 'ɜ', 'iː', 'uː', 'ɔː', 'ɑː', 'ɜː',
    # 鼻元音 (法语特有，通常带波浪号)
    'ɛ̃', 'ɑ̃', 'ɔ̃', 'œ̃', 
    # 常见的双元音组合成分
    'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'eə', 'ɪə', 'ʊə',
    # 带重音符号的西语元音
    'á', 'é', 'í', 'ó', 'ú'
}
def is_vowel_func(phone_label):
    # 1. 预处理：去掉 MFA 常见的修饰符
    clean_label = phone_label.replace('ˈ', '').replace('ˌ', '').replace(' ', '')
    
    # 2. 核心元音字符集
    vowel_chars = 'aeiouyøœɒɔɛɪʊʌæɑəɜáéíóú'

    if clean_label in VOWEL_SYMBOLS:
        return True
    if '̃' in clean_label: # 处理法语鼻元音的组合字符
        return True
    if any(c in clean_label for c in vowel_chars):
        return True
        
    return False

def process_data(tg_dir):
    data = []

    for file in os.listdir(tg_dir):
        if not file.endswith('.TextGrid'): continue
        
        try:
            tg = textgrid.openTextgrid(os.path.join(tg_dir, file), includeEmptyIntervals=False)
            phone_tier = tg.getTier('phones') 
            for entry in phone_tier.entries:
                start, end, label = entry.start, entry.end, entry.label
                
                if label in ['', 'sp', 'sil', 'spn', '<unk>']: continue
                
                duration = end - start
                if duration < 0.0001 or duration > 0.6: continue 
                is_vowel = is_vowel_func(label)
                data.append({
                    'file_id': file.replace('.TextGrid', ''),
                    'phone': label,
                    'duration': duration,
                    'type': 'vowel' if is_vowel else 'consonant'
                })
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    return pd.DataFrame(data)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Options for data processing")
    parser.add_argument(
        "--lang", 
        type=str, 
        required=True, 
        help="language, either full name or short name"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="mfa_output",
        help="where to store the wavs and metadata"
    )
    args = parser.parse_args()
    # 示例运行
    df_phones = process_data(f'{args.dir}/{args.lang}/textgrid')
    # 读取你之前的 metrics.csv
    df_metrics = pd.read_csv(f'{args.dir}/{args.lang}/alignment_analysis.csv') 
    # 合并数据
    df_final = pd.merge(df_phones, df_metrics, left_on='file_id', right_on='file')
    df_final.to_csv(f'{args.dir}/{args.lang}/stat_data.csv', index=False)

    # 提取元音数据
    vowel_durations = df_final[df_final['type'] == 'vowel']['duration']
    print(len(vowel_durations))

    # 1. 极大似然估计 (MLE)
    shape, loc, scale = stats.lognorm.fit(vowel_durations, floc=0)

    # 2. 可视化拟合效果
    plt.figure(figsize=(8, 5))
    plt.hist(vowel_durations, bins=50, density=True, alpha=0.6, color='g', label='Actual Data')
    x = np.linspace(0, 0.5, 100)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)
    plt.plot(x, pdf, 'r-', lw=2, label='Log-Normal Fit (MLE)')
    plt.title(f'Phoneme Duration Distribution & MLE Fitting of {args.lang.upper()}')
    plt.legend()
    plt.savefig(f'{args.dir}/{args.lang}/test2.jpg')
    

    print(f"Estimated Parameters: mu={np.log(scale):.4f}, sigma={shape:.4f}")
    
    pop_mean = vowel_durations.mean()
    
    # 2. 生成样本均值分布
    n = 50  # 样本量
    num_simulations = 1000
    means = [vowel_durations.sample(n).mean() for _ in range(num_simulations)]
    
    # 3. 计算样本均值的均值
    sample_means_mean = np.mean(means)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(means, bins=30, density=True, color='skyblue', alpha=0.7, 
             edgecolor='black', label='Distribution of Sample Means')
    plt.hist(vowel_durations, bins=50, density=True, alpha=0.6, color='g', label='Actual Data')
    x = np.linspace(0, 0.5, 100)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)
    plt.plot(x, pdf, 'r-', lw=2, label='Log-Normal Fit (MLE)')
    # 叠加正态拟合曲线
    mu_fit, sigma_fit = stats.norm.fit(means)
    x = np.linspace(min(means), max(means), 100)
    plt.plot(x, stats.norm.pdf(x, mu_fit, sigma_fit), 'r-', lw=2, label='Normal Fit')
    
    # 总体均值线 (红色实线)
    plt.axvline(pop_mean, color='red', linestyle='-', linewidth=1, 
                label=f'Population Mean ($\mu$): {pop_mean:.4f}')
    
    # 样本均值之均值线 (黄色虚线)
    plt.axvline(sample_means_mean, color='yellow', linestyle='--', linewidth=1, 
                label=f'Mean of Sample Means: {sample_means_mean:.4f}')
    
    plt.title(f'Verification of CLT ({args.lang.upper()}, n={n})')
    plt.xlabel('Duration (s)')
    plt.ylabel('Density')
    plt.legend()
    
    # 打印对比结果，用于报告数据
    print(f"[{args.lang}] Population Mean: {pop_mean:.6f}")
    print(f"[{args.lang}] Mean of Sample Means: {sample_means_mean:.6f}")
    print(f"[{args.lang}] Difference: {abs(pop_mean - sample_means_mean):.6f}")
    
    plt.savefig(f'{args.dir}/{args.lang}/clt_comparison.jpg')


    
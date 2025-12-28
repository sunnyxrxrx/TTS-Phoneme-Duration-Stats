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
    # 鼻元音 
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
    
    # 3. 逻辑判断：
    if clean_label in VOWEL_SYMBOLS:
        return True
    if '̃' in clean_label: # 处理法语鼻元音的组合字符
        return True
    if any(c in clean_label for c in vowel_chars):
        return True
        
    return False

def process_data(tg_dir):
    data = []
    vowels = 'aeiouəɐɛɪɒɔʊiɑɔ' 
    
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
        "--dir",
        type=str,
        default="mfa_output",
        help="where to store the wavs and metadata"
    )
    args = parser.parse_args()
    df_phones1 = process_data(f'{args.dir}/es/textgrid')
    df_metrics1 = pd.read_csv(f'{args.dir}/es/alignment_analysis.csv') 
    df_final1 = pd.merge(df_phones1, df_metrics1, left_on='file_id', right_on='file')
    df_final1.to_csv(f'{args.dir}/es/stat_data.csv', index=False)
    
    df_phones2 = process_data(f'{args.dir}/fr/textgrid')
    df_metrics2 = pd.read_csv(f'{args.dir}/fr/alignment_analysis.csv') 
    df_final2 = pd.merge(df_phones2, df_metrics2, left_on='file_id', right_on='file')
    df_final2.to_csv(f'{args.dir}/fr/stat_data.csv', index=False)


    # 提取元音数据
    vowel_durations1 = df_final1[df_final1['type'] == 'vowel']['duration']
    vowel_durations2 = df_final2[df_final2['type'] == 'vowel']['duration']

    # 1. 降采样到 n=121
    n = 121
    sample_fr = np.random.choice(vowel_durations2, n, replace=False)
    sample_es = np.random.choice(vowel_durations1, n, replace=False)

    # 2. 构造差值变量 Z
    Z = sample_fr - sample_es

    # 3. 计算统计量
    Z_bar = np.mean(Z)
    S_z = np.std(Z, ddof=1)
    T_stat = Z_bar / (S_z / np.sqrt(n))


    print(f"Z_bar: {Z_bar:.4f}")
    print(f"S_z: {S_z:.4f}")
    print(f"T-statistic: {T_stat:.4f}")

    var_fr = np.var(sample_fr, ddof=1)
    var_es = np.var(sample_es, ddof=1)
    F_stat = var_fr / var_es
    print(f"F-statistic (n=100): {F_stat:.4f}")
    
    
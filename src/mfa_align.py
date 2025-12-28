import pandas as pd
import os
import shutil
import subprocess
from tqdm import tqdm

from pypinyin import pinyin, Style
import argparse

def convert_to_mfa_pinyin(text):
    # Style.TONE3 会把 "上海" 转成 "shang4 hai3"
    # 这是 MFA 中文模型最标准的输入格式
    pinyin_list = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    
    # 过滤掉标点符号，只保留拼音
    result = [p[0] for p in pinyin_list if p[0].isalnum() or (p[0][:-1].isalnum() if p[0][-1].isdigit() else False)]
    return " ".join(result)



def prepare_mfa_data(lang_code, csv_path, wav_root, output_dir, sample_size=500):
    """
    lang_code: 语言代码如 'de', 'bg'
    csv_path: 你的 csv 文件路径
    wav_root: 音频文件存放的根目录
    output_dir: MFA 要求的输入目录
    """
    # 1. 读取并抽样
    df = pd.read_csv(csv_path, sep='|', names=['file_path', 'duration', 'text'], quoting=3)
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    lang_out = os.path.join(output_dir, lang_code)
    os.makedirs(lang_out, exist_ok=True)
    
    print(f"Processing {lang_code}...")
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        src_path = os.path.join(wav_root, row['file_path'])
        file_id = os.path.basename(row['file_path']).replace('.', '_')
        dst_wav = os.path.join(lang_out, f"{file_id}.wav")
        dst_lab = os.path.join(lang_out, f"{file_id}.lab")
        
        # 2. 转换音频格式 (MFA 最稳的格式: 16k, mono, wav)
        # 需要系统安装了 ffmpeg
        cmd = f"ffmpeg -i {src_path} -ar 16000 -ac 1 {dst_wav} -y -loglevel quiet"
        subprocess.run(cmd, shell=True)
        if lang_code=="zh":
            row['text']=convert_to_mfa_pinyin(row['text'])
        # 3. 写入文本文件
        with open(dst_lab, 'w', encoding='utf-8') as f:
            f.write(str(row['text']))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Options for data processing")
    parser.add_argument(
        "--lang", 
        type=str, 
        required=True, 
        help="language, either full name or short name"
    )
    args = parser.parse_args()
    prepare_mfa_data(args.lang, f'your/csv/address', 'your/wavs/address', './mfa_input')
# 跑完上面再跑：
# mfa align ./mfa_input/en english_mfa english_mfa ./mfa_output/en

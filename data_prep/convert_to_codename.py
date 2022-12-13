import pandas as pd
import os
import shutil

df = pd.read_csv("data_source_list.csv", encoding="latin-1")
print(df)
files_in_dir = os.listdir(".")
print(files_in_dir)
for (existing, target) in zip(df['title'], df['codename']):
    if (existing == "NaN" or existing == "playlist"):
        continue
    existing = existing.encode("ascii", errors="ignore").decode().replace(' ', '').replace('｜', '|').strip().lower()
    mp3_file = next(x for x in files_in_dir if x.replace(' ', '').replace('｜', '|').strip().lower().startswith(existing) and x.replace(' ', '').replace('｜', '|').strip().lower().endswith(".mp3"))
    vtt_file = next(x for x in files_in_dir if x.replace(' ', '').replace('｜', '|').strip().lower().startswith(existing) and x.replace(' ', '').replace('｜', '|').strip().lower().endswith(".vtt"))
    shutil.copy(mp3_file, target + ".mp3")
    shutil.copy(vtt_file, target + ".vtt")

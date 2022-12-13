# Import libraries
import pandas as pd
import math
import os
import torchaudio

# Import data 
for i in range(0,17): 
    file_name = f"vid{i}.csv"
    try:
        df = pd.read_csv(f"./mfa_testout/{file_name}")
    except:
        continue
    #df = pd.read_csv(file_name)
    df = df[df.Type == "words"]
    df = df.drop(["Type", "Speaker"], axis = 1) 

    start_time = float(df.Begin.iat[0])
    end_time = 0.0
    labels = []
    j = 0
    for idx, row in df.iterrows():
        row_start = float(row.Begin)
        if row_start < start_time + 5.0:
            labels.append(str(row.Label))
            end_time = float(row.End)
            continue

        # TODO: Use torchaudio to extract samples
        wav_filename = f"vid{i}.wav"
        wav, sr = torchaudio.load(wav_filename)
        start_sample = max(0, int(math.floor(start_time * sr)))
        end_sample = min(wav.size(dim=1), int(math.ceil(end_time * sr)))
        segment_name = f"segments/vid{i}_seg{j}"
        

        torchaudio.save(f"{segment_name}.wav", wav[:,start_sample:end_sample], sr)

        with open(f"{segment_name}.txt", 'w', encoding='utf-8') as f:
            f.write(' '.join(labels))

        start_time = row_start
        labels = [str(row.Label)]
        end_time = float(row.End)
        j += 1

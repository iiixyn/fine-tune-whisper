# Import lib
import os

# pseudo-code 
#dir = ""
#videos = [] # list the files that end with .wav in the dir

for i in range(2,17):
    start_idx = 0
    j = 0

    vid_path = f"text_to_modify/reference/vid{i}.txt"

    if not os.path.exists(vid_path):
        continue
    
    ref = ""
    with open(vid_path, 'r', encoding='utf-8') as f:
        ref = f.read()

    lower_ref = ref.lower()

    seg_path = f"text_to_modify/current/vid{i}_seg{j}.txt"
    while os.path.exists(seg_path):
        print(f"=== VID {i} SEGMENT {j} ===")
        segment = ""
        with open(seg_path, 'r', encoding='utf-8') as f:
            segment = f.read()
        segment_soup = "".join(segment.split())

        curr_idx = start_idx
        for letter in segment_soup:
            curr_idx = lower_ref.index(letter, curr_idx) + 1
        end_idx = curr_idx + 1
        
        print(ref[start_idx:end_idx].strip())
        fixed = ref[start_idx:end_idx].strip()

        with open(f"text_to_modify/good/vid{i}_seg{j}.txt", 'w', encoding='utf-8') as f:
            f.write(fixed)

        start_idx = end_idx
        j += 1
        seg_path = f"text_to_modify/current/vid{i}_seg{j}.txt"

"""
for vid in videos:
    start_idx = 0
    original = "" # read the vid.txt into a long string in lower case
    segments = [] # list the files .txt that begin with the vid name without the ext
    for seg in segments:
        letters = "" # Read the current segment
        new = "" # to append to the new segment 
        for ltr in letters:
            if ltr == " ": # to skip the spaces in the current segment
                continue
            elif ltr == original[some idx] # if the letter matches with the original, append to the new string 
                new.append(ltr) 
"""
                
                


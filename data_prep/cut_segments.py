import webvtt
import math
from datetime import datetime
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# DON'T RUN THIS IN COLAB, IT'S HARDCODED FOR LOCAL USE !!!!
# Set timestamp format
TIME_FORMAT = "%H:%M:%S.%f"

# Iterate through vid0 to vid16
for i in range(0, 17):

    # i.e. vid0.wav
    wav_filename = f"vid{i}.wav"

    # Load the wav file using torchaudio
    wav, sr = torchaudio.load(wav_filename)

    # i.e. vid0.vtt
    vtt_filename = f"vid{i}.vtt"

    # Load the captions using webvtt
    captions = webvtt.read(vtt_filename)

    # For each caption that was read,
    for j, caption in enumerate(captions):

        # Extract the start timestamp and convert it to a datetime.
        start_seconds = (datetime.strptime(caption.start, TIME_FORMAT) - datetime(1900,1,1)).total_seconds()

        # Multiply it by the wav file sample rate to get the index of the start sample.
        start_sample = max(0, int(math.floor(start_seconds * sr)))

        # Do the same for the end timestamp and sample index.
        end_seconds = (datetime.strptime(caption.end, TIME_FORMAT) - datetime(1900,1,1)).total_seconds()

        # Add 2 seconds to the end_sample because there's always sopme delay with the captions.
        # TODO: Figure out a better way to get the "right" data associated with a caption.
        # Some timestamps are really bad!! For example, vid0 is pretty terrible...
        # vid11 seems pretty good in comparison.
        end_sample = min(wav.size(dim=1), int(math.ceil(end_seconds * sr)) + 32000)

        # Create a new file in the segments folder, named vid0_seg0.wav/.txt, for example
        segment_name = f"segments/vid{i}_seg{j}"

        # Save the wav file using torchaudio; slice it using start_sample and end_sample indices in the 2nd dim.
        torchaudio.save(f"{segment_name}.wav", wav[:,start_sample:end_sample], sr)

        # Write to a txt file the text for the caption. Make sure to replace line breaks with spaces.
        with open(f"{segment_name}.txt", 'w', encoding='utf-8') as f:
            f.write(' '.join(caption.text.splitlines()))

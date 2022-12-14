# fine-tune-whisper
**Authors:**
- Xinyi Zhang
- Yimei Yang

# Main model code
## Colab links
[Newly updated Colab link](https://colab.research.google.com/drive/1WOgV7vOOJ4RTB92v1bOvUy_g3dRQNvvx?usp=sharing)

[Cleaned Colab link](https://colab.research.google.com/drive/1g5nLM8hEHJ54OCOdZRe4nUhmiUtf7nYJ#scrollTo=VmoVsORRwrIV)

[Old Colab link](https://colab.research.google.com/drive/19zGy4s6NvsjAD4l5wvdc17x1zpf_3OsJ#scrollTo=iIY4nTTQIz0d)

It is possible that read access has to requested for the Colab notebooks. 
A copy of the "updated" Colab notebook can be found below the "cleaned" Colab notebook. This contains working code about the deletion issue. 
A copy of the "cleaned" Colab notebook can be found in this repository as `fine_tune_whisper_for_QCfrench.ipynb`.

The old Colab link shows a preliminary version of the project with exploratory steps.

## Dataset
Before running the notebook, the dataset should be downloaded from [Google Drive](https://drive.google.com/drive/folders/1y3oP2XLPWqjgfrDXrdkJ8iwjH05SVmtE?usp=sharing) and placed in a directory that is usable by the notebook (e.g. `/content/...` on Colab).

In order to create a new dataset that is suited for this experiment, the data preparation procedure described in the report can be followed. The different scripts used for that preparation are detailed below.

## Notebook instructions and steps
The entire notebook can be run at once (<kbd>Ctrl</kbd>+<kbd>F9</kbd>) and should function correctly in a Debian/Ubuntu Linux-based environment with Python 3 installed. For ideal performance, a CUDA-compatible Nvidia GPU should be available on the host. However, the notebook was written on Google Colab and will function best in that environment.

### Environment set-up
These cells will install the required dependencies and output some information about the environment the notebook is running in (for example, the available GPUs that can be used).
Dependencies for this notebook include, notably:
- ffmpeg for audio processing
- torch for running the model, torchaudio for audio processing
- numpy, pandas and sklearn for data processing
- Huggingface Transformers for optimizers and schedulers
- tensorflow for TensorBoard usage
- pytorch_lightning for training modules and TensorBoard logging
- tqdm for prettifying output
- ray(tune) for training tuning
- evaluate for WER/CER metrics
- whisper, the model that we are fine-tuning

### Import libraries
Here, the Python notebook environment is setup with the required modules and libraries. It is adapted for usage in Colab, so it might need some tweaking to be used outside of it.

### Set up data and helper functions
In this section, the dataset directory is specified, as well as multiple constants and helper functions. One would need to change the `DATASET_DIR` if using a custom dataset or running the notebook without having the same Google Drive directory structure. If correctly downloaded, the dataset should have 1569 WAV files and 1569 TXT files in it.

The sample rate of the WAV files should be 16 kHz. If not, the notebook will resample them to 16 kHz regardless when using the `load_wave` helper function (potentially causing quality loss).

The French tokenizer is loaded from the Whisper model, using its "small" model with the "fr" language. Those parameters can be changed to fine-tune a smaller or larger version of the model.

`get_audio_file_list` will create pairs of WAV and TXT files based on their filenames, and will also limit the length of the audio files if they exceed 6 seconds (96000 samples) and the text files if they exceed 1200 characters. This is to ensure consistent behavior when training the fine-tuned model.

The dataset is then split into a training, testing and validation set. The testing set is 20% of the original dataset, and the validation set is 25% of the training set. The training set thus represents the remaining 60% of the dataset. The testing set is only used at the end of the model to verify performance and calculate Word Error Rate (WER). The validation set is used throughout training to evaluate the training performance of the model and fine-tune the learning rate.

The `FrSpeechDataset` class is a `torch.utils.data.Dataset` using the `audio_file_list` created before and adapting it into a format that is usable by Whisper. Whisper expects items to be in a dictionary with 3 fields:
- `input_ids` contains the Log-Mel Spectrogram of the audio file;
- `labels` contains the text prompts, encoded with the tokenizer and finished by an end-of-text token;
- `dec_input_ids` contains the text prompts, encoded with the tokenizer and beginning with a special start-of-text token sequence (see `tokenizer.sot_sequence_including_no_timestamps`).

The `WhisperDataCollatorWithPadding` class is a data processing class that transforms the previously mentioned dictionary into batches. The batches are equally padded (to have the same length for all items) with constant values that have special encodings (-100 and 50257), corresponding to "skip" or "end-of-text" tokens.

The data is then loaded through `DataLoader`s to be available in batches.

Finally, some examples of the data provided by the loaders is available at the end of the section. The special tokens such as `<|startoftranscript|><|fr|><|transcribe|><|notimestamps|>` and `<|endoftext|>` can be seen.

### Define the fine-tuning model
First, a configuration is created in the `Config` class. First-attempt hyperparameters (`learning_rate`, `weight_decay`...) are defined there based on heuristics for the field (based on similar research or recommended by our mentor).

The fine-tuned model is created as the `WhisperModelModule` class. This class extends `LightningModule`, allowing it to use the `pytorch-lightning` model structure to facilitate quick implementation and runtime.

The standard Whisper decoding options, base model and tokenizer are used. However, **all the parameters of the model's encoder are frozen**:
```python
# only decoder training
for p in self.model.encoder.parameters():
    p.requires_grad = False
```

This makes it so that only the decoder part of the model will be trained (thus, fine-tuning it). An additional `CrossEntropyLoss` loss function is added to specifically train the decoder using that metric. The config of the model as well as its training and evaluation datasets are setup there as well.

The `forward` function of the model is the same as that of base Whisper.

The `training_step` function is slightly modified so that the encoder is not trained and the `loss_fn` function is calculated.

The `validation_step` function is modified to calculate the CER (Character Error Rate) and WER (Word Error Rate) metrics, as well as outputting progress bars and logging.

The `configure_optimizers` function sets up the `AdamW` optimizer used for taining. It also sets up a scheduler to organize the training steps with some warmup.

The `setup`, `train_dataloader` and `val_dataloader` functions are used to setup the model using the previously defined dataloaders and configuration.

### Main
This is the main section of the notebook, where things can be run (training, eval) after being defined in the previous section.

First, some parameters are defined to specify which Whisper model is to be used, and where training logs and artifacts will be saved. The torch CUDA cache is emptied so that GPU memory is reset between runs, and the garbage collector is called to maximize available RAM (this is important when running the model multiple times on Colab).

Then, the config defined above is created, and a TensorBoardLogger is created to have a visual dashboard for the training process. A callback function is defined so that a checkpoint of the model is saved after every epoch. The model is also created using `WhisperModelModule` and the previously defined datasets.

The model is then run using a pytorch-lightning `Trainer`. This object sets up TensorBoard logging and checkpoint callbacks, as well as some other parameters.

The trainer is then used in tuning mode to find a suitable learning rate. In our case, the tuner suggested a learning rate of 0.00012022644346174131, for example (minimized loss without going into diverging loss).

The `Config` class is then adjusted with this new learning rate as well as batch size and training epoch count. Those values can be adjusted depending on the capabilities of the training environment (time, computing resources). The model is recreated using that new config.

Finally, `trainer.fit(model)` will train the fine-tuned Whisper model.

### Visualize the model performance throughout training
This section displays training/validation metrics through TensorBoard. Some available metrics are the following:
- Learning rate
- Training loss
- Validation CER per epoch
- Validation CER per step
- Validation loss per epoch
- Validation loss per step
- Validation WER per epoch
- Validation WER per step

### Best model
This section gets some results from the best model (in the notebook at that point, it was checkpoint 0009) and the baseline (not fine-tuned) Whisper model. It gets the labels (ground truth) and predictions using the testing dataset.

### Baseline WER
This section is used to calculate the Word Error Rate (WER) of the **baseline**, which is the Whisper model as-is (without fine-tuning). It also shows some examples to get an idea of the errors in the model's output.

### Our model's WER
This section does the same but with the fine-tuned Whisper model. It also shows some examples so that the changes can be observed in the output.

# Data preparation code
## clean_n_cut.py
This script reads audio-prompt alignment data exported from MFA (Montreal Forced Aligner, see section below) in CSV format, and uses that data to segment the associated WAV file into smaller, ~5 seconds segments.

The script is hard-coded to read up to 17 files named `vid0.csv`, `vid1.csv`... `vid16.csv` from the `./mfa_testout/` relative path. The associated `vid0.wav`, `vid1.wav`... `vid16.wav` files must be in the same directory as the script itself (relative path).

For each input pair (CSV and WAV), the script will segment it according to MFA alignment/transcription timestamps. Each segment is represented as a `segments/vid{i}_seg{j}.wav` file, where `i` is the `vid` file index and `j` is the current segment counter (it increments after each segment is created), and an associated `segments/vid{i}_seg{j}.txt` file containing the matching transcription from MFA. Note that the transcription data created by MFA strips a lot of relevant sentence markers (capital letters, punctuation, ...) and needs to be transformed through the use of `correct_text.py` afterwards.

## convert_to_codename.py
This script converts the audio and captions files listed in the hand-crafted `data_source_list.csv` into an anonymized/standardized version. For example, it will convert WAV and VTT files starting with `La semaine verte | La COVID de la tomate` into `vid3.mp3` and `vid3.vtt`. It strips special characters present in the source filename when considering equality.

## correct_text.py
This script is used to correct the segments created by `clean_n_cut.py` into a version matching the original (cleaned-up) captions file. This helps ensure that fine-tuning Whisper with the segments data does not make the model "forget" about punctuation and sentence structure!

The script is hard-coded to read reference captions from `text_to_modify/reference/vid{i}.txt` (relative path, `i` is the file index from 0 to 16). It then reads each segment output from `clean_n_cut.py` (matching the `i` index) sequentially, and attempts to match the location of the segment in the original captions file. To do so, it checks every alphanumeric character from the segment (MFA strips punctuation and other marks, but not whitespace) and attempts to find the first index of it in the reference file (put in lowercase). When it does find a character, it tracks its index and keeps going with the rest of the reference file. When a segment is completed, it keeps one more character to take into account eventual punctuation marks, such as a comma or period (it strips whitespace, however) It keeps going until all segments for a given `vid` file have been matched.

Here is an example of how this script functions:
```
# MFA Output
Segment 0: "cet après-midi le lapin"
Segment 1: "dormait au soleil il aime"
Segment 2: "bien se coucher sur le tapis"

# Reference Caption
"Cet après-midi, le lapin dormait au soleil. Il aime bien se coucher sur le tapis dans le bureau."

# Segment 0: starting from index 0, match "c", then match "e", "t", "a"... "n". Result: Indices 0 to 24
# Match: "Cet après-midi, le lapin"
# Segment 1: starting from index 25, match "d", then match "o", "r"... "e". Result: Indices 26 to 52 (idx 25 is whitespace)
# Match: "dormait au soleil. Il aime"
# Segment 2: starting from index 53, match "b", then match "i", "e"... "s". Result: Indices 54 to 82 (idx 53 is whitespace)
# Match: "bien se coucher sur le tapis"
# ...
```

## cut_segments.py
This script is a remnant of some original tests to split the captions files (VTT) into usable audio and text segments. It had no alignment capabilities and resulted in a poor quality dataset. Do not use this script, it is only there for documentation purposes.

## data_source_list.csv
This hand-crafted CSV file contains the original video name, URL, and naming convention for each file used in the dataset. It also contains the duration in minutes and seconds of each original video file (for statistics purposes).

# Montreal Forced Aligner
In order to increase the quality of the text labels used to create the training dataset, Montreal Forced Aligner (MFA) was used. This open-source project can be installed by following the instructions on [their documentation website](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html). It allows "forced alignment" of text prompts to a given audio file, creating millisecond-aligned timestamps in the process. For example, given the prompt "les lapins aiment manger des bananes" and a corresponding audio WAV file, MFA will output the following in CSV format:
| Begin | End   | Label   |
|-------|-------|---------|
| 0.000 | 0.123 | les     |
| 0.123 | 0.789 | lapins  |
| 0.789 | 1.234 | aiment  |
| 1.234 | 1.999 | manger  |
| 1.999 | 2.111 | des     |
| 2.111 | 2.500 | bananes |

This output is read and processed in the `clean_n_cut.py` script, see sections above.

To process the audio and text files in the same way that was used to create the dataset in this project, the following MFA commands have to be run:
```sh
# Download the acoustic French language model
mfa models download acoustic french_mfa

# Download the French language dictionary
mfa models download dictionary french_mfa

# Align the files present in the ./mfa_test directory (relative path), using the french_mfa acoustic model and dictionary.
# The output will be multiple CSV files (one for each WAV-TXT pair) in the ./mfa_testout directory.
# The original text is preserved as much as possible (--include_original_text), but this doesn't really work (see sections above).
# Pause duration is reduced to 0.001 (default is 0.05) as Quebec French has a fast speech rate and short inter-word pauses, lots of coarticulation.
mfa align --output_format csv ./mfa_test french_mfa french_mfa ./mfa_testout --include_original_text --min_pause_duration 0.001 --clean
```

# Quick-Voice-Conversion

My implementation of [QuickVC](https://github.com/quickvc/QuickVC-VoiceConversion), a deep learning based model for any-to-many voice conversion.

The [original implementation](https://github.com/quickvc/QuickVC-VoiceConversion) is a bit messy thus hard to extend, this version is cleaner and well-documented, providing a cleaner base for further development.

## Environment

Tested on Ubuntu22, RTX 4090, Nvidia driver version 560.35.03, CUDA Version 12.6.
These are not necessary to match, but when there is a problem these can be refered to.

However, a CUDA-enabled GPU is needed to run the scripts since we use operations like `.cuda()` directly.
If you want to run without GPU please adapt the script for non-GPU.

The environment of this repo is structured using [UV](https://docs.astral.sh/uv/), a modern python package manager.
Install UV following instructions on [this page](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it on your machine.

Following the following steps to create the environment:

```bash
uv sync # Create a .venv containing all necessary packages
uv pip install -e . # Install src as package

# Activate env
# For Ubuntu
source .venv/bin/activate
# For Windows
.venv/Script/activate.bat # (maybe, not sure, something like this)
```

## Example Usage

We provide an experiment using [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) as an example to showcase the usage.

### Prepare audio files

We rely on `jvs_wav_preprocessed` containing folders of audio files of each speaker, which looks like follows:

```txt
jvs_wav_preprocessed
|--jvs001
|   |--BASIC5000_0025.wav
    |--...
|--jvs002
    |--...
|--...
```

There are two reference files (`datasets/jvs/jvs_preprocessed_{train/val}.txt`) containing a list of files in `jvs_wav_preprocessed`, which are used to refer to audio files for training.

Note that in the reference files, the root path for audios is set to `data/jvs`. If you wish to use them directly, please place the preprocessed dataset `jvs_wav_preprocessed` under `data/jvs`.
If you decided to use your own reference files, please change the file paths for reference files in the training config file, e.g., in `configs/qvc_hubert.json`, change `data/training_files` and `data/validation_files` to your own file path.

### Preprocess audio files

QVC relies on pretrained models for speech content feature extraction for more details refer to [their paper](https://arxiv.org/abs/2302.08296).
Since these pretrained models will not be updated during training, we extract their extracted features before hand to avoid runtime processing during training for speeding up. These extracted featurse will only be loaded during training.

To do this, run the following command to preprocess:

```bash
python scripts/preprocess/preprocess_weo.py --wav-dirpath data/jvs/jvs_wav_preprocessed --mode hubert
```

Prameters:

- `--wav-dirpath`: all files end with `.wav` in this directoty will be processed. This is done recursively for all sub-directories.
- `--mode`: choose which pretrained model to use. We currently support `hubert` and `whisper_large_v2`. You can add your own by modifying this script. For detailed usage please refer to the script.

### Training

Run the following command to train a model:

```bash
python scripts/train/train.py --config configs/qvc_hubert.json --model qvc_hubert
```

Parameters:

- `--config`: the path of the config file.
- `--model`: the name of the log file. The log will be stored to `logs/${model}`, in this case, `logs/qvc_hubert`.

Note that this script is for multi-node training, i.e., it automatically uses all GPUs for training.
If you only want to use some of them, add environment variable before the command as follows:

```bash
CUDA_VISABLE_DEVICES=0 python scripts/train/train.py -c configs/qvc_hubert.json -m qvc_hubert
```

### Inference

We provide a script to convert a list of audios to their corresponding target audio via a txt file.
An example can be found in `assets/example/convert.txt`.
You can adapt this file for you own batched conversion.

We also provide a (not fully) pretrained checkpoint based on hubert which can be downloaded from [this url](https://drive.google.com/file/d/1F-w8k7YBETHeelXmSDUniJzkeLMRjT5h/view?usp=share_link).
Please unzip the checkpoint and place it under `logs` as `logs/qvc_hubert`.

Convert audios using the following command:

```bash
python scripts/infer/convert.py \
--hpfile logs/qvc_hubert/config.json \
--ptfile logs/qvc_hubert/G_126000.pth \
--txtpath assets/example/convert.txt \
--outdir outputs/example/converted
```

Parameters:

- `--hpfile`: the config file for the checkpoint you want to use.
- `--ptfile`: the checkpoint file for the generator you want to use.
- `--txtpath`: the txt file containing audios you want to convert as well as their target audios.
- `--outdir`: the directory where you want to store the converted audios.

## Acknowledgement

Heavily rely on [QVC](https://github.com/quickvc/QuickVC-VoiceConversion).
Cite QVC using the following BibTxt:

```txt
@inproceedings{guo2023quickvc,
  title={Quickvc: A lightweight vits-based any-to-many voice conversion model using istft for faster conversion},
  author={Guo, Houjian and Liu, Chaoran and Ishi, Carlos Toshinori and Ishiguro, Hiroshi},
  booktitle={2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={1--7},
  year={2023},
  organization={IEEE}
}
```

## Contact

Feel free to open an issue.

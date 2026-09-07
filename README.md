# Multimodal Deepfake Detection

This project is about detecting deepfake videos using both **audio and visual information**.

I started with the idea of Audio-Visual Feature Fusion (AVFF) and have been working on adapting the implementation for my own experiments, mainly around dataset preparation, local training, evaluation, and running the pipeline on GPU.

The main goal is to understand how audio and video features can be combined to improve deepfake detection instead of relying only on individual video frames.

## What this project does

The pipeline works with two types of information from a video:

- Video frames
- Audio

These features are processed separately and then combined to learn an audio-visual representation, which is used for deepfake classification.

At the moment, most of my work is focused on getting the complete pipeline running locally and making the data preparation and evaluation easier to work with.

## Project structure

```text
Deepfake-Detection/
│
├── src/                         # Model and data processing code
├── egs/                         # Training scripts
├── data/                        # Dataset metadata / CSV files
│
├── dfdc_extractor.py            # DFDC data extraction
├── dfdc_extractor_local.py      # Local data extraction
├── generate_csv.py              # Generate dataset CSV files
├── eval.py                      # Run evaluation
├── get_metrics.py               # Calculate evaluation metrics
│
├── adapt_pretraining_weights.ipynb
├── reconstruction_demo.ipynb
├── dfdc_vids.txt
├── requirements.txt
└── README.md

```

*Note: The actual dataset, checkpoints, and experiment outputs are not committed to this repository because of their size.*

## Dataset

I am currently working with the DeepFake Detection Challenge (DFDC) dataset format. The dataset itself is not included here. After downloading the dataset, the paths need to be updated according to the local setup.

Some of the scripts in the repository are specifically used for extracting the required video/audio information and preparing the CSV files used by the training and evaluation pipeline.

## Setup

I am using a Linux environment with an NVIDIA GPU for the experiments.

Create a conda environment:

```bash
conda create -n deepfake python=3.8 -y
conda activate deepfake

```

Install the required packages:

```bash
pip install -r requirements.txt

```

Check whether PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found')"

```

You can also check the NVIDIA GPU directly with:

```bash
nvidia-smi

```

## Preparing the data

The dataset needs to be processed before training.
For local extraction:

```bash
python dfdc_extractor_local.py

```

The CSV files can then be generated using:

```bash
python generate_csv.py

```

*(The exact paths depend on where the DFDC dataset is stored.)*

## Training

The training setup is divided into three stages.

**Stage 1:** The first stage is used for representation pretraining.

```bash
cd egs
bash stage-1.sh

```

**Stage 2:** The second stage continues the pretraining/adaptation using real-face videos.

```bash
cd egs
bash stage-2.sh

```

**Stage 3:** The final stage is used for the deepfake classification task.

```bash
cd egs
bash stage-3.sh

```

*Note: The batch size and other training parameters may need to be changed depending on the GPU being used.*

## Evaluation

After training, a checkpoint can be evaluated using:

```bash
python eval.py \
    --checkpoint path/to/checkpoint.pth \
    --csv_file data/testset.csv

```

Metrics can then be calculated using:

```bash
python get_metrics.py

```

## GPU experiments

One of the main reasons I am working on this project locally is to experiment with the training pipeline on an NVIDIA GPU. For example, before starting a training run I usually check available memory with `nvidia-smi`. Training settings such as batch size can be adjusted based on available GPU memory.

## My work on this project

I modified the original Audio-Visual Feature Fusion implementation to handle severe class imbalances and local hardware constraints. Key modifications include:

* **Data Balancing:** Rewrote dataset generation logic (`generate_csv.py`) to mitigate extreme class imbalance (1,248 Fake vs. 86 Real videos) via strict majority undersampling.
* **Real-Time Augmentation:** Integrated dynamic video frame augmentations (color jittering, contrast adjustments, random cropping) into the PyTorch `dataloader.py` to prevent overfitting on the minority class.
* **Hardware Optimization:** Resolved CUDA Out-of-Memory (OOM) limits on local hardware by optimizing data loaders and restricting the batch size (`batch_size=1`) for an 11.5GB chunk of the DFDC dataset.
* **Pipeline Automation:** Adapted evaluation scripts to track Binary Cross Entropy and Contrastive Loss metrics during constrained local training.

## Results

By implementing the data balancing and real-time augmentation pipeline, the model's ability to learn generalized features improved significantly, preventing it from blindly predicting the majority class.

| Setup | Dataset | GPU | Peak mAP | Peak AUC |
| --- | --- | --- | --- | --- |
| Baseline (Imbalanced) | DFDC (11.5GB Chunk) | NVIDIA GPU | - | 0.380 |
| Modified (Balanced + Augmentation) | DFDC (11.5GB Chunk) | NVIDIA GPU | 0.635 | 0.715 |

*Note: Due to hardware-induced batch size constraints (batch_size=1), metrics exhibit expected stochastic fluctuation across epochs. The modified peak was achieved during Epoch 2 of a constrained 5-epoch run.*

## Demo

There is also a notebook for looking at the reconstruction part of the model:
`reconstruction_demo.ipynb`

This is useful for getting a better idea of what the representation learning stage is doing.

## Reference

This project is based on ideas from:
**AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection**
Original paper: https://arxiv.org/abs/2406.02951

The original implementation was useful as the starting point for this project. This repository contains my own modifications and experiments around that implementation to adapt it for highly constrained local environments and imbalanced subsets.

```

```

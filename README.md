
# Gaussian Adaptive Attention-Based Speech Network for Depression Detection

This repository contains the code and models for the paper ["Gaussian Adaptive Attention-based Speech Network: Enhancing Feature Understanding for Mental Health Disorders"](https://arxiv.org/abs/2024.XXXXXX). The repository extends the original work presented in the EUSIPCO 2021 paper, ["Gender Bias in Depression Detection Using Audio Features"](https://arxiv.org/abs/2010.15120), and introduces two advanced models, DAAMAudioCNNLSTM and DAAMAudioTransformer, designed for audio-based depression detection.

## Table of Contents

- [Credit](#credit)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Notes](#notes)
- [References](#references)
- [License](#license)

## Credit

This repository is based on the original implementation from ["Gender Bias in Depression Detection Using Audio Features"](https://arxiv.org/abs/2010.15120) by Bailey et al. The code has been modified to include new model architectures and configurations for enhanced performance and explainability in depression detection using audio data.

## Overview

Depression detection from speech data is a challenging task due to the subtle nature of depressive symptoms in vocal patterns and the inherent data scarcity. In this project, we introduce two novel models:

1. **DAAMAudioCNNLSTM**: A hybrid CNN-LSTM model enhanced with a Gaussian Adaptive Attention Mechanism (GAAM) to dynamically focus on the most informative parts of the speech data.
2. **DAAMAudioTransformer**: A transformer-based model incorporating the GAAM module, designed to capture both local and global dependencies in audio data.

These models are evaluated on the DAIC-WOZ dataset and demonstrate state-of-the-art performance in detecting depression from speech signals, achieving F1 macro scores of 0.702 and 0.72, respectively.

## Prerequisites

The project was developed using Python 3.x and is optimized for Ubuntu 18.04. It is recommended to use a conda environment to manage dependencies.

### Install Miniconda and Create Environment

1. **Install Miniconda**: [Miniconda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. **Create the Environment**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment**:
   ```bash
   conda activate myenv
   ```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/gaussian_attention_depression_detection.git
   cd gaussian_attention_depression_detection
   ```

2. **Prepare the Environment**:
   Ensure you have Python 3.x installed and set up a conda environment using the provided `environment.yml` file.

3. **Install Dependencies**:
   Install the necessary Python packages by activating the conda environment as described in the Prerequisites section.

## Usage

### Running the Training Script

To train the models, use the `main1.py` script with the appropriate arguments:

```bash
python main1.py train --validate --cuda --vis --position=1
```

### Evaluating the Models

To evaluate the trained models on the validation or test sets:

```bash
python main1.py test --validate --cuda --vis --position=1
```

Arguments:
- `train` or `test`: Specify whether to train or test the model.
- `--validate`: Use the validation set during training or testing.
- `--cuda`: Enable GPU acceleration.
- `--vis`: Visualize training progress and results.
- `--position=1`: Specify which config file to use.

## Dataset

The DAIC-WOZ dataset is used for training and evaluating the models. This dataset is part of the DARPA DCAPS program and can be obtained through The University of Southern California. The dataset contains audio recordings, transcriptions, and other multimodal data, but this work focuses primarily on the audio component.

**Dataset Preprocessing**:
- **Mel-Spectrograms**: Extracted using a Hanning window with a Mel filterbank that includes 40 frequency bins. Features are normalized using z-normalization.
- **Raw Audio**: Similar preprocessing steps are applied, ensuring consistency across different input representations.

More details on dataset preprocessing are provided in the original [pre-processing framework](https://github.com/adbailey1/daic_woz_process).

## Models

### DAAMAudioCNNLSTM

- **Architecture**: Combines CNN and LSTM layers with a GAAM module for attention-based feature extraction.
- **Attention Mechanism**: Gaussian distributions are used to dynamically focus on the most relevant parts of the input speech data.

### DAAMAudioTransformer

- **Architecture**: A transformer-based model incorporating the GAAM module.
- **Attention Mechanism**: Similar to DAAMAudioCNNLSTM but uses transformer encoders to capture both local and global dependencies.

## Results

### Performance Metrics

Both models achieve state-of-the-art performance on the DAIC-WOZ dataset:

- **DAAMAudioCNNLSTM**: F1 macro score of 0.702.
- **DAAMAudioTransformer**: F1 macro score of 0.72.

These results highlight the models' effectiveness in detecting depression from speech data without relying on supplementary information like speaker details or vowel positions.

## Notes

- The models are designed to be explainable, providing insights into the decision-making process through the GAAM module.
- Future work will explore the integration of textual and visual data into these models to further enhance their diagnostic capabilities.

## References

Please refer to the original papers for more detailed explanations and methodologies:

- Bailey, A., et al. (2021). Gender Bias in Depression Detection Using Audio Features. *EUSIPCO 2021*. [arXiv:2010.15120](https://arxiv.org/abs/2010.15120)
- Ioannides, G., et al. (2024). Gaussian Adaptive Attention-based Speech Network: Enhancing Feature Understanding for Mental Health Disorders. *IEEE Transactions on Affective Computing*. [arXiv:2024.XXXXXX](https://arxiv.org/abs/2024.XXXXXX)

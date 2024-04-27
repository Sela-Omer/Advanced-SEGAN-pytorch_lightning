# SEGAN: Speech Enhancement Generative Adversarial Network - Pytorch Lightning Implementation

## Overview

SEGAN (Speech Enhancement Generative Adversarial Network) is a state-of-the-art deep learning model designed to enhance the quality of speech audio signals by reducing background noise and improving clarity. This implementation leverages the Pytorch Lightning framework to facilitate scalable and efficient training, making it suitable for both research and production environments. The project is structured to be modular and customizable, allowing for easy experimentation and adaptation to different audio processing needs.

## Installation

### Prerequisites

Ensure you have the following prerequisites installed on your system before proceeding with the installation of the SEGAN project:

- Python 3.10 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the Repository**: First, clone this repository to your local machine using Git:

   ```bash
   git clone git@github.com:Spittoon/SEGAN-pytorch.git
   cd segan
   ```

2. **Install Dependencies**: Install all the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
To start the project, you can execute the `__init__.py` file directly from the command line:

```bash
python __init__.py
```

### Training

To train the model execute it with the APP_mode config option set to FIT.

### Evaluation

To evaluate the model execute it with the APP_mode config option set to EVAL.

### Configuration

The `config.ini` parameters can be dynamically overridden at runtime by passing them as command line arguments to the `__init__.py` file. This allows for flexible adjustments of parameters without permanently changing the configuration file.\
Example Command to Override Settings:
```bash
python __init__.py --APP_mode FIT --APP_arch SEGAN
```

## Project Execution Flow

This section outlines the systematic flow of operations from initialization to training or evaluation of the SEGAN model.

### 1. Initialization (`__init__.py`)

- The project starts in the `__init__.py` file located in the main project directory. This script serves as the entry point and is responsible for setting up the initial environment and configurations.
- It parses command line arguments which can override the default settings specified in `config.ini`, allowing dynamic adjustments of parameters such as batch size and environment mode.

### 2. Service Layer (`src/service`)

- After initialization, control is passed to the service layer, where specific services are defined to handle different aspects of the application:
  - **service.py**: Abstract base class defining essential service functionalities.
  - **service_fit.py**: Inherits from `service.py`, setting up everything needed for the training process.
  - **service_eval.py**: Also inherits from `service.py`, tailored for handling the evaluation process.

### 3. Script Execution (`src/script`)

- Scripts in the `src/script` directory utilize the services to perform specific tasks:
  - **segan_fit_script.py**: Uses `service_fit.py` to configure and start the training of the SEGAN model.
  - **segan_eval_script.py**: Uses `service_eval.py` for evaluating the model's performance on a test dataset.
- These scripts are directly executable and provide the interface for interacting with the model training or evaluation phases.

### 4. Model Setup (`src/module`)

- Within the model setup:
  - **segan.py**: Central module integrating the generator and discriminator components of the SEGAN model.
  - **segan_generator.py** and **segan_discriminator.py**: Define the architecture of the generator and discriminator, respectively.
- The model files are imported and utilized by the service scripts to instantiate the SEGAN model, set up loss functions, optimizers, and other training or evaluation components.

### 5. Training/Evaluation

- **Training (Train Mode)**:
  - If the project is run in train mode (`MODE=FIT`), `segan_fit_script.py` orchestrates the entire training cycle, managing data loading, model training iterations, and saving checkpoints.
- **Evaluation (Eval Mode)**:
  - If the project is run in eval mode (`MODE=EVAL`), `segan_eval_script.py` handles the loading of a pre-trained model and performs evaluations on new data to gauge the model's performance.

This flow ensures that each component of the SEGAN project is optimally utilized to achieve the best results in enhancing speech audio quality through noise reduction and clarity improvement.


## Project Structure

This section provides a comprehensive overview of each directory and file within the `src` directory, explaining their purpose and functionality in detail.

### src/config

- **config.ini**: Central configuration file containing settings for model parameters, training environment, and operational modes.
- **config.py**: Script that parses `config.ini` and exposes these settings as Python variables throughout the project.

### src/datamodule

- **audio_data_module.py**: Manages the setup, loading, and preprocessing of audio datasets for training, validation, and testing phases using PyTorch Lightning.

### src/dataset

- **audio_dataset.py**: Basic class for handling audio files, supports reading and preprocessing of data.
- **iter_audio_dataset.py**: Iterable dataset class for streaming large audio datasets efficiently.
- **map_audio_dataset.py**: Map-style dataset class providing flexible data access.

### src/display

- **display_waveform.py**: Contains functions for plotting audio waveforms, aiding in the visual analysis of audio signals.

### src/helper

- **audio_helper.py**: Utility functions for audio data manipulation.
- **metrics_helper.py**: Functions for calculating and managing performance metrics.
- **param_helper.py**: Helper for parameter management and retrieval.

### src/immutable

- **data_index.py**: Defines immutable data indices for dataset handling.
- **progress_bar.py**: Custom progress bar for displaying training and processing progress.
- **xy_data_index_pair.py**: Defines pairs of data indices for handling complex data mappings.

### src/module

- **segan.py**: Main module for the SEGAN model, integrating components like generator and discriminator.
- **segan_discriminator.py**: Defines the discriminator part of the SEGAN model.
- **segan_generator.py**: Defines the generator part of the SEGAN model.
- **virtual_batch_norm.py**: Implements virtual batch normalization used in the SEGAN model.

### src/script

- **eval_script.py**: Script for conducting model evaluations.
- **fit_script.py**: Script for running model training sessions.
- **script.py**: Template for basic script setup.
- **segan_eval_script.py**: SEGAN-specific script for model evaluation.
- **segan_fit_script.py**: SEGAN-specific script for model training.
- **segan_script.py**: General script for SEGAN model operations.

### src/service

- **service.py**: Abstract base class for defining core service functionalities.
- **service_eval.py**: Extends `service.py`, tailored for evaluation processes.
- **service_fit.py**: Extends `service.py`, tailored for training processes.

## Configuration Settings

The `config.ini` file contains several sections that define the settings and parameters for various aspects of the SEGAN project. Here's a detailed breakdown of each section and its options:

### [APP]
- **ENVIRONMENT**: Defines the operational environment. Options include DEVELOPMENT and PRODUCTION. Development mode might enable additional logging or debugging features that are not available in production.
- **MODE**: Specifies the operational mode of the model. Options include FIT for training and EVAL for evaluation.
- **ARCH**: Specifies the model architecture. For this project, it is set to SEGAN.
- **BATCH_SIZE**: Determines the number of samples processed before the model's internal parameters are updated. Currently set to 400.
- **DATA_SUBSET_SIZE_PERCENT**: Specifies the percentage of the dataset to be used during training or evaluation. Set to 1.0, meaning the entire dataset is used.
- **CPU_WORKERS**: Number of CPU workers used for loading data. Set to 11 to optimize data loading operations.
- **ACCELERATOR**: Specifies the hardware accelerator to use. Set to 'auto' to automatically choose the best available option, typically GPU if available.
- **MODEL_STORE_PATH**: The directory path where trained models are stored. Set to 'model'.
- **DEVICES**: Number of devices to use for training. Set to 1.
- **NUM_NODES**: Number of nodes for distributed training. Set to 1.
- **STRATEGY**: Training strategy to manage distributed or single device training. Set to 'auto'.

### [FIT]
- **N_EPOCHS**: The number of complete passes through the training dataset. Currently set to 120.
- **TORCH_PRECISION**: The computational precision setting for PyTorch operations. Options include 'high' for higher precision operations.
- **LOG_EVERY_N_STEPS**: Specifies how often to log training progress. Set to 1, indicating logging at every training step.
- **CHECKPOINT_MONITOR**: A list of metrics to monitor for checkpointing during training. Includes various audio quality metrics like PESQ, CSIG, CBAK, COVL, and SSNR.
- **LOG_GRAPH**: Boolean indicating whether to log the computational graph of the model during training. Set to True.
- **MODEL_HYPERPARAMS**: Hyperparameters specific to the model components such as generators and discriminators. Learning rates for the generator (lr_gen) and discriminator (lr_disc) are both set to 0.0001.

### [DATA]
- **PATH**: Specifies the path to the dataset used for training and validation. This is essential for locating the data files needed for model operations.

### [EVAL]
- **CHECKPOINT_PATH**: Specifies the path to the checkpoint file used for model evaluation. This setting is crucial for resuming training or for performing evaluation using a pre-trained model. Currently set to 'model/SEGAN/version_4/checkpoints/SEGAN-epoch=02-valid_CBAK=1.33.ckpt'.

## References and Acknowledgments

This project is based on the original SEGAN (Speech Enhancement Generative Adversarial Network) paper. The paper can be accessed through the following link:

- Pascual, S., Bonafonte, A., & Serra, J. (2017). SEGAN: Speech Enhancement Generative Adversarial Network. Retrieved from [arXiv:1703.09452](https://arxiv.org/abs/1703.09452).

This implementation of the SEGAN project incorporates various metrics and methods adapted from the `pysepm` library, which is a Python implementation of speech quality evaluation metrics. These metrics are essential for evaluating the performance of the speech enhancement model.

The specific functions and methodologies were adapted to fit the architectural and functional requirements of this project, ensuring that the SEGAN model performs optimally under diverse conditions.

For more detailed information on the metrics used and their implementation, please refer to the `pysepm` repository:

- [pysepm on GitHub](https://github.com/schmiph2/pysepm)
# Advanced SEGAN: Speech Enhancement Generative Adversarial Network - Pytorch Lightning Implementation

## Read Our Article

To gain a comprehensive understanding of the methodologies and enhancements applied in our Advanced SEGAN project, please refer to our detailed Medium article: [Advancing SEGAN: Proposing and Evaluating Architectural Improvements](https://medium.com/@Omer_Sela/advancing-segan-proposing-and-evaluating-architectural-improvements-3e367c71758a).

## Overview

Advanced SEGAN builds upon the original SEGAN (Speech Enhancement Generative Adversarial Network) model, introducing significant improvements in architecture and training processes. This enhanced version leverages Pytorch Lightning to facilitate scalable and efficient training, suitable for both research and practical applications. The modifications are aimed at improving the clarity and quality of speech audio signals by reducing background noise more effectively than the original model.

Attached is a table of the metrics showcasing the performance of our Advanced SEGAN Architecture.

|Metric|Noisy|Wiener|Original SEGAN|Our Basic SEGAN|Advanced SEGAN|
|:---:|:---:|:---:|:---:|:---:|:---:|
|PESQ|1.97|2.22|2.16|2.17|2.34|
|CSIG|3.35|3.23|3.48|2.93|3.5|
|CBAK|2.44|2.68|2.94|2.99|3.19|
|COVL|2.63|2.67|2.8|2.5|2.92|
|SSNR|1.68|5.07|7.73|9.9|9.75|
‘Noisy’, ‘Weiner’ and ‘Original SEGAN’ were taken from `Pascual, S., Bonafonte, A., & Serra, J. (2017). SEGAN: Speech Enhancement Generative Adversarial Network. Retrieved from [arXiv:1703.09452](https://arxiv.org/abs/1703.09452)`.

## Installation

### Prerequisites

Ensure you have the following prerequisites installed on your system before proceeding with the installation of the SEGAN project:

- Python 3.10 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the Repository**: First, clone this repository to your local machine using Git:

   ```bash
   git clone git@github.com:Sela-Omer/Advanced-SEGAN-pytorch_lightning.git
   cd SEGAN-pytorch_lightning
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

The execution flow of the Advanced SEGAN project is designed for streamlined operation from initialization through training or evaluation:

### 1. Initialization (`__init__.py`)
This step initializes the project, setting up the environment and configurations based on `config.ini` or command line arguments.

### 2. Service Layer (`src/service`)
This layer manages functionalities for different operational modes:
- **service_fit.py**: Configures the training process.
- **service_eval.py**: Manages the evaluation process.

### 3. Script Execution (`src/script`)
Scripts execute specific tasks using the services defined:
- **adv_segan_script.py**: Executes advanced model operations, enhancing both standard and new functionalities.
- **segan_fit_script.py**: Directs traditional training operations.
- **segan_eval_script.py**: Manages the traditional model evaluation.

### 4. Model Setup (`src/module`)
Incorporates all model components:
- **adv_segan.py**: Advanced SEGAN model integration.
- **segan.py**: Basic SEGAN model integration.
- **segan_generator.py** and **segan_discriminator.py**: Define the architectures of the generator and discriminator.

### 5. Training/Evaluation
- **Training**: Managed under `FIT` mode, directing data handling, model training, and saving state via advanced scripts.
- **Evaluation**: Conducted under `EVAL` mode, assessing model performance on test datasets.

This structured approach ensures efficient resource use and optimal component functioning for superior speech enhancement outcomes.


## Project Structure

This section details the organization of the Advanced SEGAN project, explaining the functionality and purpose of each component within the directory structure:


#### `src/config`
- **config.ini**: Central configuration file with editable settings for model operations.
- **config.py**: Parses and provides access to settings in `config.ini`.

#### `src/datamodule`
- **audio_data_module.py**: Manages audio data handling for the model.

#### `src/dataset`
- **audio_dataset.py**: Basic audio data handling class.
- **iter_audio_dataset.py**: Iterable dataset for large-scale audio data processing.
- **map_audio_dataset.py**: Map-style dataset class for efficient data access.

#### `src/display`
- **display_waveform.py**: Tools for visualizing audio data.

#### `src/helper`
- **audio_helper.py**: Utility functions for audio processing.
- **metrics_helper.py**: Functions for computing performance metrics.
- **param_helper.py**: Helper class for managing model parameters.

#### `src/immutable`
- **data_index.py**: Defines constant indices used across datasets.
- **progress_bar.py**: Customizable progress bar for tracking model training and evaluations.
- **xy_data_index_pair.py**: Manages pairs of indices for complex data relationships.

#### `src/module`
- **adv_segan.py**: Advanced SEGAN model implementation.
- **res_block_1d.py**: Implementation of 1D residual blocks.
- **segan.py**: Standard SEGAN model implementation.
- **segan_discriminator.py**: Discriminator part of the SEGAN model.
- **segan_generator.py**: Generator part of the SEGAN model.
- **segan_residual_generator_bn.py**: SEGAN generator with residual blocks and batch normalization.
- **transposed_res_block_1d.py**: Transposed 1D residual blocks for the model.
- **virtual_batch_norm.py**: Implements virtual batch normalization technique.

#### `src/script`
- **adv_segan_eval_script.py**: Script for evaluating the Advanced SEGAN model.
- **adv_segan_fit_script.py**: Script for training the Advanced SEGAN model.
- **adv_segan_script.py**: General script for running the Advanced SEGAN model operations.
- **eval_script.py**: General evaluation script for models.
- **fit_script.py**: General training script for models.
- **script.py**: Template for basic script operations.
- **segan_eval_script.py**: SEGAN-specific evaluation script.
- **segan_fit_script.py**: SEGAN-specific training script.
- **segan_script.py**: Script for general SEGAN model operations.
- **stats_script.py**: Script for generating statistical data from model operations.

#### `src/service`
- **service.py**: Base class for defining service operations.
- **service_eval.py**: Service class for handling model evaluations.
- **service_fit.py**: Service class for managing training operations.
- **service_stats.py**: Service class for statistical analysis of model performance.

## Configuration Settings

The `config.ini` file contains various sections that define the settings and parameters for different aspects of the SEGAN project. Here's a breakdown of each section and its options:

### [APP]
- **ENVIRONMENT**: Specifies the operational environment for the model.
- **MODE**: Determines the mode of operation such as training or evaluation.
- **ARCH**: Defines the architecture of the model.
- **BATCH_SIZE**: Sets the number of samples processed in one batch.
- **DATA_SUBSET_SIZE_PERCENT**: Percentage of the dataset used during training or evaluation.
- **CPU_WORKERS**: Number of CPU workers used for data loading.
- **ACCELERATOR**: Hardware acceleration options (e.g., GPU).
- **MODEL_STORE_PATH**: Directory where model checkpoints are saved.
- **DEVICES**: Number of devices used for training.
- **NUM_NODES**: Number of nodes for distributed training.
- **STRATEGY**: Strategy for distributing training across devices.

### [FIT]
- **N_EPOCHS**: Total number of training epochs.
- **TORCH_PRECISION**: Computational precision for training operations.
- **LOG_EVERY_N_STEPS**: Frequency of logging training progress.
- **CHECKPOINT_MONITOR**: Metrics monitored for model checkpointing.
- **LOG_GRAPH**: Whether to log the model's computational graph.
- **MODEL_HYPERPARAMS**: Specific hyperparameters for the model.
- **TRAINER_PRECISION**: Precision setting for the training process.

### [DATA]
- **PATH**: Directory path where the dataset is stored.
- **NOISY_MEAN**: Mean value of noisy input data.
- **NOISY_STD**: Standard deviation of noisy input data.
- **CLEAN_MEAN**: Mean value of clean target data.
- **CLEAN_STD**: Standard deviation of clean target data.

### [EVAL]
- **CHECKPOINT_PATH**: Path to the model checkpoint used for evaluation.

## References and Acknowledgments

This project is based on the original SEGAN (Speech Enhancement Generative Adversarial Network) paper. The paper can be accessed through the following link:

- Pascual, S., Bonafonte, A., & Serra, J. (2017). SEGAN: Speech Enhancement Generative Adversarial Network. Retrieved from [arXiv:1703.09452](https://arxiv.org/abs/1703.09452).

This implementation of the SEGAN project incorporates various metrics and methods adapted from the `pysepm` library, which is a Python implementation of speech quality evaluation metrics. These metrics are essential for evaluating the performance of the speech enhancement model.\
The specific functions and methodologies were adapted to fit the architectural and functional requirements of this project, ensuring that the SEGAN model performs optimally under diverse conditions.\
For more detailed information on the metrics used and their implementation, please refer to the `pysepm` repository:

- [pysepm on GitHub](https://github.com/schmiph2/pysepm)
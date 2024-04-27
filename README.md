# SEGAN: Speech Enhancement Generative Adversarial Network Pytorch Lightning Implementation

## Overview

This project implements the Speech Enhancement Generative Adversarial Network (SEGAN), which is designed to improve the
quality of audio signals through deep learning techniques. The SEGAN model aims to reduce noise and enhance the clarity
of spoken audio.

## Installation

Before starting, ensure you have Python installed on your system. To install all required dependencies, navigate to the
project directory and run the following command:

pip install -r requirements.txt

## Project Structure

### src/config

This directory contains the configuration files for the project. It includes:

**config.ini**: A configuration file in INI format that specifies global parameters and settings used across various
modules of the project.\
**config.py**: A Python script that reads and parses the config.ini file, making configuration settings available as
Python variables.

### src/datamodule

The datamodule directory houses modules that manage the loading and preprocessing of audio data:

**audio_data_module.py**: Defines a PyTorch Lightning data module for handling audio datasets, facilitating the easy
setup of train, validation, and test splits along with necessary preprocessing steps.

### src/dataset

This directory includes classes and functions for managing audio datasets:

**audio_dataset.py**: Provides a basic PyTorch dataset class for audio files, enabling file reading and preprocessing.\
**iter_audio_dataset.py**: Implements an iterable PyTorch dataset for streaming audio data, useful for very large
datasets.\
**map_audio_dataset.py**: A map-style PyTorch dataset that provides more flexible data access patterns than the iterable
version.

### src/display

Modules for visual representation of audio data:

**display_waveform.py**: Contains functions to plot audio waveforms, helping in the visual analysis of audio signals
before and after processing.

### src/helper

Utility modules that provide additional functionalities:

**audio_helper.py**: Includes functions to assist with audio processing tasks such as loading audio files and converting
formats.\
**metrics_helper.py**: Contains implementations of various metrics for evaluating audio quality and the performance of
the SEGAN model.\
**param_helper.py**: Helps in parsing and managing hyperparameters from configuration files or command line arguments.

### src/immutable

Contains classes and utilities that provide immutable data structures:

**data_index.py**: Manages indices for data samples, ensuring they remain constant throughout the project execution.\
**progress_bar.py**: A custom progress bar utility for displaying training progress in a user-friendly manner.\
**xy_data_index_pair.py**: Defines a data structure for storing paired data samples and their corresponding indices,
ensuring data integrity and traceability.

### src/module

Core modules of the SEGAN model:

**segan.py**: The main module that integrates the SEGAN model components including the generator and discriminator
networks.\
**segan_discriminator.py**: Defines the discriminator part of the GAN, responsible for distinguishing between real and
enhanced audio samples.\
**segan_generator.py**: Implements the generator network that attempts to enhance input noisy audio samples.\
**virtual_batch_norm.py**: Provides an implementation of virtual batch normalization, used within the generator and
discriminator to stabilize training.

### src/script

Scripts for training and deploying the SEGAN model:

**fit_script.py**: A script that setups and runs the training loop for the SEGAN model.\
**segan_fit_script.py**: An extension to fit_script.py that includes additional configurations and custom setups
specific to SEGAN training.

### src/service

Services related to model deployment and operation:

**service.py**: Defines basic service functions for deploying the SEGAN model in a production environment.\
**service_fit.py**: Contains functions tailored to fit the model to specific service requirements, enabling fine-tuning and
optimization for deployment.

## Configuration Details (config.ini)

The config.ini file organizes various settings and parameters needed for the operation of the SEGAN model. Each section
of the configuration file is detailed below:

### [APP]

**ENVIRONMENT**: Defines the mode of operation, such as DEVELOPMENT or PRODUCTION. Development mode might enable
additional logging or debugging features not present in production. \
**MODE**: Specifies the operation mode of the model, such as FIT for training or EVAL for evaluation.\
**ARCH**: Indicates the architecture of the model, in this case, SEGAN.\
**BATCH_SIZE**: The number of samples processed before the model's internal parameters are updated, set to 400. \
**DATA_SUBSET_SIZE_PERCENT**: Specifies the percentage of the dataset to be used, set to 1.0, meaning the entire dataset
is used.  
**CPU_WORKERS**: Number of CPU workers used for loading data, set to 11.\
**ACCELERATOR**: Specifies the hardware accelerator to use, such as auto to automatically choose the best available option (GPU, if available).\
**MODEL_STORE_PATH**: The directory path where trained models are stored, here set to model.\
**DEVICES**: Number of devices to use, set to 1.\
**NUM_NODES**: Number of nodes for distributed training, set to 1.\
**STRATEGY**: Training strategy, auto to automatically manage distributed or single device training.

### [FIT]

**N_EPOCHS**: The number of complete passes through the training dataset, set to 120.\
**TORCH_PRECISION**: The computational precision setting for PyTorch, such as high for higher precision operations.\
**LOG_EVERY_N_STEPS**: How often to log training progress, set to 1, indicating logging at every training step.\
**CHECKPOINT_MONITOR**: A list of metrics to monitor for checkpointing, including various audio quality metrics like
PESQ, CSIG, CBAK, COVL, and SSNR.\
**LOG_GRAPH**: Boolean indicating whether to log the computational graph, set to True.\
**MODEL_HYPERPARAMS**: Hyperparameters specific to the model components like generators and discriminators, such as
learning rates lr_gen and lr_disc, both set to 0.0001.

### [DATA]

**PATH**: The path to the dataset used for training and validation, set to data.
import os
import zipfile
from urllib import request

import lightning as pl
import torch.utils.data
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from tqdm import tqdm

from src.helper.audio_helper import read_audio_metadata
from src.immutable.data_index import DataIndex
from src.immutable.progress_bar import ProgressBar
from src.immutable.xy_data_index_pair import XyDataIndexPair
from src.service.service import Service


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, service: Service, data_dir, audio_dataset_class, sample_rate=16000):
        """
        Initializes the AudioDataModule.

        Parameters:
            data_dir (str): The directory where the data is stored.
            audio_dataset_class: The class representing the audio dataset.
            sample_rate (int, optional): The sample rate of the audio data. Defaults to 16000.
        """
        super().__init__()
        self.service = service
        self.audio_dataset_class = audio_dataset_class
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.bitstream_prefix = 'http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942'

        self.Xy_data_index_lst = [
            XyDataIndexPair(
                X_data_index=DataIndex(url=f'{self.bitstream_prefix}/noisy_trainset_wav.zip', id='noisy'),
                y_data_index=DataIndex(url=f'{self.bitstream_prefix}/clean_trainset_wav.zip', id='clean'),
                stage='train',
            ),
            XyDataIndexPair(
                X_data_index=DataIndex(url=f'{self.bitstream_prefix}/noisy_testset_wav.zip', id='noisy'),
                y_data_index=DataIndex(url=f'{self.bitstream_prefix}/clean_testset_wav.zip', id='clean'),
                stage='val',
            ),
        ]

    def prepare_data(self):
        """
        Prepares the data by downloading and extracting it.

        This function iterates through each `Xy_data_index_pair` in the `Xy_data_index_lst` list and for each `data_index` in the pair, it downloads and extracts the data from the specified URL and directory. The downloaded and extracted data is then converted to WAV format using the `convert_wavs` method.

        Parameters:
            self (AudioDataModule): The instance of the AudioDataModule class.

        Returns:
            None
        """
        for Xy_data_index_pair in tqdm(self.Xy_data_index_lst):
            for data_index in [Xy_data_index_pair.X_data_index, Xy_data_index_pair.y_data_index]:
                self.download_and_extract(data_index.url, os.path.join(self.data_dir, data_index.dir))
                self.convert_wavs(os.path.join(self.data_dir, data_index.dir))

    def setup(self, stage=None):
        """
        Sets up the data for the given stage of the experiment.

        Parameters:
            stage (str, optional): The stage of the experiment. Defaults to None.

        Returns:
            None
        """
        find_index_pair_by_stage = lambda stage: \
            [index_pair for index_pair in self.Xy_data_index_lst if index_pair.stage == stage][0]

        valid_index_pair = find_index_pair_by_stage('val')
        valid_dataset = self._subset_dataset(self.audio_dataset_class(valid_index_pair, data_dir=self.data_dir,
                                                                      target_sample_rate=self.sample_rate,
                                                                      window_size_overlap_percentage=0.0))

        if stage == 'fit' or stage is None:
            train_index_pair = find_index_pair_by_stage('train')
            self.train_dataset = self._subset_dataset(self.audio_dataset_class(train_index_pair, data_dir=self.data_dir,
                                                                               target_sample_rate=self.sample_rate,
                                                                               window_size_overlap_percentage=0.5))
            self.valid_dataset = valid_dataset
        if stage == 'test' or stage == 'predict':
            self.predict_dataset = valid_dataset
            self.test_dataset = valid_dataset

    def _subset_dataset(self, dataset):
        """
        Subsets the given dataset based on the dataset_size_percent attribute of the service.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to be subset.

        Returns:
            torch.utils.data.Subset: The subset of the input dataset.

        """
        # If the dataset_size_percent is greater than or equal to 1, return the input dataset as is.
        if self.service.dataset_size_percent >= 1:
            return dataset

        # Calculate the subset size based on the dataset_size_percent attribute.
        subset_size = int(len(dataset) * self.service.dataset_size_percent)

        # Return a subset of the input dataset using the calculated subset size.
        return torch.utils.data.Subset(dataset, range(subset_size))

    def download_and_extract(self, url, extract_to):
        """
        Downloads and extracts a file from the given URL to the specified directory.

        Parameters:
            url (str): The URL of the file to download and extract.
            extract_to (str): The directory where the file will be extracted to.

        Returns:
            None
        """
        zip_path = f'{extract_to}.zip'
        if not os.path.exists(zip_path):
            print(f'ZIP PATH  {zip_path} DOES NOT EXIST. DOWNLOADING DATASET FROM {url}...')
            zip_dir = os.path.dirname(zip_path)
            if not os.path.exists(zip_dir):
                os.makedirs(zip_dir)
            request.urlretrieve(url, zip_path, ProgressBar())
        if not os.path.exists(extract_to):
            print(f'INFLATING ZIP FROM {zip_path} TO {extract_to} ...')
            self.extract_ext_files_flat(zip_path, extract_to, extension='.wav')

    def extract_ext_files_flat(self, zip_path, output_folder, extension='.wav'):
        """
        Extracts all files with a specific extension from a zip file and saves them
        in a specified output folder.

        Args:
            zip_path (str): Path to the zip file.
            output_folder (str): Path to the output folder.
            extension (str, optional): The extension of the files to extract. Defaults to '.wav'.
        """
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the zip
            for file in zip_ref.namelist():
                # Check if the file ends with the specific extension
                if file.endswith(extension) and not file.endswith('/'):
                    # Read the file data
                    with zip_ref.open(file) as file_data:
                        data = file_data.read()
                    # Write the data to a new file in the output folder
                    # Extracting the filename from the path to avoid directory creation
                    base_filename = os.path.basename(file)
                    output_file_path = os.path.join(output_folder, base_filename)
                    with open(output_file_path, 'wb') as new_file:
                        new_file.write(data)

    def convert_wavs(self, wav_dir):
        """
        Converts WAV files in a directory to a desired sample rate.

        This function iterates through each WAV file in the specified directory. If the sample rate of a WAV file is not the same as the desired sample rate, it resamples the audio to the desired sample rate and saves it back to the same location.

        Args:
            wav_dir (str): The directory containing the WAV files.

        Returns:
            None
        """

        # List to store the paths of files that need to be converted
        files_to_convert = []

        # Iterate through each file in the directory
        for wav_file in os.listdir(wav_dir):
            # Check if the file is a WAV file
            if wav_file.endswith('.wav'):
                # Get the full path of the WAV file
                wav_path = os.path.join(wav_dir, wav_file)

                # Load the audio file metadata
                metadata = read_audio_metadata(wav_path)

                # If the sample rate of the audio file is not the same as the desired sample rate, add it to the list
                if metadata.sample_rate != self.sample_rate:
                    files_to_convert.append(wav_path)

        # If no files need to be converted, return
        if len(files_to_convert) == 0:
            return

        # Print the number of files to be converted
        print(
            f"Converting WAV files to the desired sample rate... length of files to convert: {len(files_to_convert)}...")

        # Iterate through each file that needs to be converted
        for wav_path in files_to_convert:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(wav_path)

            # If the sample rate is not the desired sample rate, resample the audio
            if sample_rate != self.sample_rate:
                # Create a resampler object with the original and desired sample rates
                resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)

                # Resample the waveform
                waveform = resampler(waveform)

                # Update the sample rate to the new rate
                sample_rate = self.sample_rate

                # Save the resampled audio back to the same location
                torchaudio.save(wav_path, waveform, sample_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.service.batch_size, shuffle=True,
                          num_workers=self.service.cpu_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.service.batch_size, shuffle=False,
                          num_workers=self.service.cpu_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.service.batch_size, shuffle=False,
                          num_workers=self.service.cpu_workers,
                          persistent_workers=True)

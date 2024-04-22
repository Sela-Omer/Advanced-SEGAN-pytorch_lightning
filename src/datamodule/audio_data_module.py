import os
import subprocess
import zipfile
from urllib import request

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.immutable.data_index import DataIndex
from src.immutable.progress_bar import ProgressBar
from src.immutable.xy_data_index_pair import XyDataIndexPair


class AudioDataModule(LightningDataModule):
    def __init__(self, data_dir, audio_dataset_class, sample_rate=16000):
        """
        Initializes the AudioDataModule.

        Parameters:
            data_dir (str): The directory where the data is stored.
            audio_dataset_class: The class representing the audio dataset.
            sample_rate (int, optional): The sample rate of the audio data. Defaults to 16000.
        """
        super().__init__()
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
                self.convert_wavs(os.path.join(self.data_dir, data_index.dir),
                                  os.path.join(self.data_dir, data_index.dir))

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
        valid_dataset = self.audio_dataset_class(valid_index_pair, data_dir=self.data_dir,
                                                 target_sample_rate=self.sample_rate,
                                                 window_size_overlap_percentage=0.0)

        if stage == 'fit' or stage is None:
            train_index_pair = find_index_pair_by_stage('train')
            self.train_dataset = self.audio_dataset_class(train_index_pair, data_dir=self.data_dir,
                                                          target_sample_rate=self.sample_rate,
                                                          window_size_overlap_percentage=0.5)
            self.valid_dataset = valid_dataset
        if stage == 'test' or stage == 'predict':
            self.predict_dataset = valid_dataset
            self.test_dataset = valid_dataset

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
            os.makedirs(extract_to)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

    def convert_wavs(self, source_dir, target_dir):
        """
        Converts WAV files from a source directory to a target directory.

        Args:
            source_dir (str): The path to the source directory containing the WAV files.
            target_dir (str): The path to the target directory where the converted WAV files will be saved.

        Returns:
            None

        Raises:
            FileNotFoundError: If the source directory does not exist.
            subprocess.CalledProcessError: If the conversion process fails.

        Prints:
            A message indicating the conversion process.

        Notes:
            - Creates the target directory if it does not exist.
            - Converts each WAV file in the source directory to the target directory using the SoX library.
            - The conversion process is performed using the 'sox' command-line tool.
            - The sample rate for the conversion is determined by the 'sample_rate' attribute of the current object.
        """
        if not os.path.exists(target_dir):
            print(f'CONVERTING WAVS FROM {source_dir} TO {target_dir}')
            os.makedirs(target_dir)
            for wav_file in os.listdir(source_dir):
                if wav_file.endswith('.wav'):
                    source_path = os.path.join(source_dir, wav_file)
                    target_path = os.path.join(target_dir, wav_file)
                    subprocess.run(['sox', source_path, '-r', self.sample_rate, target_path], check=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=400, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=400, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=400, shuffle=False)

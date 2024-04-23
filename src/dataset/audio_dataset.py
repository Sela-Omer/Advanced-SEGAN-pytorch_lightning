import torch
import torchaudio

from src.helper.audio_helper import read_audio_metadata
from src.immutable.xy_data_index_pair import XyDataIndexPair


class AudioDataset:
    """
    A PyTorch Dataset that creates windows of audio data from a list of X and y files.

    Args:
        Xy_data_index_pair (XyDataIndexPair): The pair of X and y data indices.
        data_dir (str): The directory where the data is stored.
        window_size (int, optional): The size of the audio window. Defaults to 16384.
        window_size_overlap_percentage (float, optional): The overlap percentage of the window size. Defaults to 0.5.
        target_sample_rate (int, optional): The target sample rate of the audio data. Defaults to 16000.

    """

    def __init__(self, Xy_data_index_pair: XyDataIndexPair, data_dir: str, window_size=16384,
                 window_size_overlap_percentage=0.5, target_sample_rate=16000):
        """
        Initializes the AudioDataset.

        Parameters:
            Xy_data_index_pair (XyDataIndexPair): The pair of X and y data indices.
            data_dir (str): The directory where the data is stored.
            window_size (int, optional): The size of the audio window. Defaults to 16384.
            window_size_overlap_percentage (float, optional): The overlap percentage of the window size. Defaults to 0.5.
            target_sample_rate (int, optional): The target sample rate of the audio data. Defaults to 16000.

        Returns:
            None
        """
        self.data_dir = data_dir
        self.Xy_data_index_pair = Xy_data_index_pair
        self.Xy_file_pairs = Xy_data_index_pair.list_file_pairs(data_dir)
        self.target_sample_rate = target_sample_rate
        self.window_size = window_size
        self.window_overlap = int(window_size_overlap_percentage * window_size)
        self.Xy_file_frame_lst = self._create_file_frame_lst()

    def _create_file_frame_lst(self):
        """
        Creates a list of dictionaries containing information about each pair of X and y files in the dataset.

        Returns:
            list: A list of dictionaries, where each dictionary contains the following keys:
                - 'file_index' (int): The index of the file pair in the dataset.
                - 'X_file' (str): The path to the X file.
                - 'y_file' (str): The path to the y file.
                - 'start_frame' (int): The starting frame index for the file pair.
                - 'end_frame' (int): The ending frame index for the file pair.
        """
        Xy_file_frame_lst = []
        frame_count = 0
        for i, (X_file, y_file) in enumerate(self.Xy_file_pairs):
            X_num_frames = self._read_audio_numframes(X_file)
            y_num_frames = self._read_audio_numframes(X_file)
            assert X_num_frames == y_num_frames, f'Expected same number of frames but got {X_num_frames} and {y_num_frames} for files {X_file} and {y_file}'
            Xy_file_frame_lst.append(
                {'file_index': i, 'X_file': X_file, 'y_file': y_file, 'start_frame': frame_count,
                 'end_frame': frame_count + X_num_frames})
            frame_count += X_num_frames
        return Xy_file_frame_lst

    def _read_audio(self, file):
        """
        Reads an audio file and returns the waveform and sample rate.

        Parameters:
            file (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the waveform (torch.Tensor) and the sample rate (int).

        Raises:
            AssertionError: If the sample rate of the audio file does not match the target sample rate.
        """
        waveform, sample_rate = torchaudio.load(file)
        assert sample_rate == self.target_sample_rate, f'Expected sample rate of {self.target_sample_rate} but got {sample_rate} for file {file}'
        return waveform.squeeze(0), sample_rate

    def _read_audio_numframes(self, file):
        """
        Reads an audio file and returns the number of frames.

        Parameters:
            file (str): The path to the audio file.

        Returns:
            int: The number of frames in the audio file.

        Raises:
            AssertionError: If the sample rate of the audio file does not match the target sample rate.
        """
        metadata = read_audio_metadata(file)
        sample_rate = metadata.sample_rate
        num_frames = metadata.num_frames
        assert sample_rate == self.target_sample_rate, f'Expected sample rate of {self.target_sample_rate} but got {sample_rate} for file {file}'
        return num_frames

    def _conform_to_window_size(self, waveform):
        """
        Conforms the waveform to the window size.

        Parameters:
            waveform (torch.Tensor): The waveform to conform.

        Returns:
            torch.Tensor: The conformed waveform.
        """
        assert waveform.shape[
                   -1] <= self.window_size, f'Expected waveform to be less than or equal to {self.window_size} but got {waveform.shape[-1]}'
        assert len(waveform.shape) == 1, f'Expected waveform to be 1D but got {len(waveform.shape)}'

        if waveform.shape[-1] < self.window_size:
            waveform = torch.nn.functional.pad(waveform, (0, self.window_size - waveform.shape[-1]))
        return waveform[None]

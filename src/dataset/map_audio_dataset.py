import torch
from torch.utils.data import IterableDataset, Dataset

from src.dataset.audio_dataset import AudioDataset


class MapAudioDataset(AudioDataset, Dataset):
    def __len__(self):
        """
        Returns the length of the dataset by calculating the number of windows that can be created from the audio files.

        Returns:
            int: The length of the dataset.
        """
        start_frame, end_frame = 0, self.Xy_file_frame_lst[-1]['end_frame']
        return len(range(start_frame, end_frame, self.window_size - self.window_overlap))

    def _get_framedata_by_frame_index(self, index):
        """
        Retrieves the frame data corresponding to the given frame index.

        Parameters:
            index (int): The frame index for which to retrieve the frame data.

        Returns:
            dict: The frame data corresponding to the given frame index.

        Raises:
            ValueError: If the given index is out of range.
        """
        for frame_data in self.Xy_file_frame_lst:
            if frame_data['start_frame'] <= index < frame_data['end_frame']:
                return frame_data
        raise ValueError(f"Index {index} is out of range.")

    def _read_Xy_waveforms_from_framedata(self, framedata):
        """
        A function that reads X and y waveforms from the given framedata.

        Parameters:
            framedata (dict): A dictionary containing 'X_file' and 'y_file' keys specifying the file paths.

        Returns:
            tuple: A tuple containing the X_waveform (torch.Tensor) and y_waveform (torch.Tensor).
        """
        X_file, y_file = framedata['X_file'], framedata['y_file']
        X_waveform, X_sample_rate = self._read_audio(X_file)
        y_waveform, y_sample_rate = self._read_audio(y_file)
        assert X_waveform.shape[-1] == y_waveform.shape[
            -1], f'Expected same number of samples but got {X_waveform.shape[-1]} and {y_waveform.shape[-1]} for files {X_file} and {y_file}'
        return X_waveform, y_waveform

    def __getitem__(self, index):
        """
        Returns a tuple of torch.Tensor objects representing the X and y waveform windows
        for a given frame index. The windows are created by appending waveform segments
        from the corresponding X and y files until the window size is reached. The
        waveform segments are sliced based on the frame index and the start and end
        frames of each file.

        Parameters:
            frame_index (int): The index of the frame from which to start creating the windows.

        Returns:
            tuple: A tuple containing two torch.Tensor objects representing the X and y
                   waveform windows respectively.
        """
        X_window_lst, y_window_lst = [], []
        frame_index = index * (self.window_size - self.window_overlap)
        frame_count = 0
        while frame_count < self.window_size:
            try:
                framedata = self._get_framedata_by_frame_index(frame_index + frame_count)

                X_waveform, y_waveform = self._read_Xy_waveforms_from_framedata(framedata)

                waveform_start_frame = (frame_index + frame_count) - framedata['start_frame']
                waveform_end_frame = min(framedata['end_frame'] - framedata['start_frame'],
                                         waveform_start_frame + (self.window_size - frame_count))

                X_window_lst.append(X_waveform[..., waveform_start_frame:waveform_end_frame])
                y_window_lst.append(y_waveform[..., waveform_start_frame:waveform_end_frame])
                frame_count += waveform_end_frame - waveform_start_frame
            except ValueError:
                break
        return torch.cat(X_window_lst, dim=0), torch.cat(y_window_lst, dim=0)
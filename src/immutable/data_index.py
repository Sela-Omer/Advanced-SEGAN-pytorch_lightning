import os
from dataclasses import dataclass


@dataclass
class DataIndex:
    url: str
    id: str

    @property
    def dir(self):
        """
        Returns the directory name of the URL by splitting the URL string and extracting the last element.
        The directory name is obtained by splitting the last element of the split URL string by the '.' character
        and taking the first element of the resulting list.

        Returns:
            str: The directory name of the URL.
        """
        return self.url.split('/')[-1].split('.')[0]

    def list_files(self, data_dir: str):
        """
        Lists the WAV files in the given data directory.

        Args:
            data_dir (str): The path to the data directory.

        Returns:
            list: A sorted list of file paths to WAV files in the data directory.
        """
        return sorted([os.path.join(data_dir, self.dir, f) for f in os.listdir(os.path.join(data_dir, self.dir)) if
                       f.endswith('.wav')])

import os
from dataclasses import dataclass

from src.immutable.data_index import DataIndex


@dataclass
class XyDataIndexPair:
    X_data_index: DataIndex
    y_data_index: DataIndex
    stage: str

    def list_file_pairs(self, data_dir: str):
        """
        Lists the file pairs in the given data directory.

        Args:
            data_dir (str): The path to the data directory.

        Returns:
            list: A list of tuples containing the file pairs.
        """
        X_files = self.X_data_index.list_files(data_dir)
        y_files = self.y_data_index.list_files(data_dir)
        file_pair_lst = list(zip(X_files, y_files))
        for X_file, y_file in file_pair_lst:
            assert os.path.basename(X_file) == os.path.basename(y_file), f'X: {X_file} != y: {y_file}'
        return file_pair_lst
import torchaudio
from torchaudio import AudioMetaData


def read_audio_metadata(file: str) -> AudioMetaData:
    """
    Reads the metadata of an audio file.

    Args:
        file (str): The path to the audio file.

    Returns:
        AudioMetaData: The metadata of the audio file.
    """
    # Use torchaudio to read the audio file metadata
    return torchaudio.info(file)

from typing import Union


def convert_param_to_type(s: str) -> Union[int, float, str]:
    """
    Convert a string to an integer, float, or string.

    Args:
        s (str): The input string to convert.

    Returns:
        Union[int, float, str]: The converted value.

    Raises:
        None

    """
    # Try to convert the string to an integer
    try:
        return int(s)
    # If conversion to integer fails, try to convert to float
    except ValueError:
        try:
            return float(s)
        # If conversion to float fails, return the string as is
        except ValueError:
            return s

import argparse

from src.config import config
from src.service.service_eval import ServiceEval
from src.service.service_fit import ServiceFit
import torch


def parse_arguments(config):
    """
    Generate parser for command line arguments based on config.ini sections and options.

    Args:
        config (configparser.ConfigParser): The configparser object containing the configuration.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    """
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Override config.ini settings")

    # Iterate over the sections and options in the config
    for section in config.sections():
        for key in config[section]:
            # Generate a command line argument for each config setting
            arg_name = f"--{section}_{key}"
            help_msg = f"Override {section} {key}"
            parser.add_argument(arg_name, type=str, help=help_msg)

    # Parse the command line arguments
    return parser.parse_args()


def override_config(config, args):
    """
    Override config.ini settings with any specified command line arguments.

    Args:
        config (ConfigParser): The config parser object.
        args (Namespace): The command line arguments.

    Returns:
        None
    """
    # Iterate over the command line arguments
    for arg_key, arg_value in vars(args).items():
        # Check if the argument value is not None
        if arg_value is not None:
            # Split the argument key to get the section and key
            arg_lst = arg_key.split('_')
            found_arg = False
            for i in range(1, len(arg_lst)):
                a1 = '_'.join(arg_lst[:i])
                a2 = '_'.join(arg_lst[i:])
                if config.has_option(a1, a2):
                    config.set(a1, a2, arg_value)
                    found_arg = True
            if not found_arg:
                raise ValueError(f"Invalid command line argument was passed. {arg_key} does not exist in config.ini")


if __name__ == "__main__":
    print('listing available cuda devices:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    args = parse_arguments(config)
    override_config(config, args)

    app_mode = config['APP']['MODE']
    app_arch = config['APP']['ARCH']

    service_dict = {
        "FIT": ServiceFit,
        "EVAL": ServiceEval,
    }
    service = service_dict[app_mode](config)

    script = service.scripts[app_arch]
    script()

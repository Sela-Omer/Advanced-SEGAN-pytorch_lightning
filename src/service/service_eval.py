from typing import Callable, Dict

from src.script.segan_eval_script import SEGAN_EvalScript
from src.service.service import Service


class ServiceEval(Service):

    def __init__(self, config):
        """
        Initializes a new instance of the ServiceEval class.

        Args:
            config (dict): A dictionary containing the configuration settings for the service.

        Returns:
            None
        """
        super(ServiceEval, self).__init__(config)

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Fit Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'SEGAN': SEGAN_EvalScript(self)  # Instantiate the SEGAN_Fit class
        }

        return script_dict

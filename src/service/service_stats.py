from typing import Callable, Dict

from src.script.segan_eval_script import SEGAN_EvalScript
from src.script.stats_script import StatsScript
from src.service.service import Service


class ServiceStats(Service):

    @property
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.

        The dictionary contains a single key-value pair.
        The key is the name of the ARCH , and the value is an instance of the Script class.
        """
        # Create a dictionary with the ARCH name and its corresponding instance
        script_dict = {
            'SEGAN': StatsScript(self),
        }

        return script_dict

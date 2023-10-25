import warnings
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


class AdditionalSearchPathPlugin(SearchPathPlugin):
    """This plugin allows to overwrite the ahcore configurations without needed to fork the repository."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        additional_path = Path(__file__).parent.parent.parent / "additional_config"
        if additional_path.is_dir():
            if not list(additional_path.glob("*")):
                warnings.warn(
                    f"Found additional_config folder in {additional_path}, without any configuration files. "
                    "If you want to overwrite the default ahcore configs, "
                    "please add these to the additional_config folder. "
                    "You can symlink your additional configuration to this folder. "
                    "See the documentation at https://docs.aiforoncology.nl/ahcore/configuration.html "
                    "for more information."
                )
            else:
                # Add additional search path for configs
                logger.info(f"Adding additional search path for configs: file://{additional_path}")
                search_path.prepend(provider="hydra-ahcore", path=f"file://{additional_path}")


def register_additional_config_search_path() -> None:
    """
    Register the additional_config folder as a search path for hydra.

    Returns
    -------
    None
    """
    Plugins.instance().register(AdditionalSearchPathPlugin)

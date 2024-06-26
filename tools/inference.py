import dotenv
import hydra
from omegaconf import DictConfig

dotenv.load_dotenv(override=True)

from ahcore.hydra_plugins import register_additional_config_search_path  # noqa: E402

register_additional_config_search_path()


@hydra.main(
    config_path="../config",
    config_name="inference.yaml",
    version_base="1.3",
)
def main(config: DictConfig) -> None:
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from ahcore.entrypoints import inference
    from ahcore.utils.io import extras, print_config, validate_config

    # Validate config -- Fails if there are mandatory missing values
    validate_config(config)

    # Applies optional utilities
    extras(config)

    if config.get("print_config"):
        print_config(config, resolve=True)

    # Train model
    return inference(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

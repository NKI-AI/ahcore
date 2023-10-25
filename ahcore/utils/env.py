import dotenv


def load_env() -> None:
    """
    Load environment variables from `.env` file if it exists.
    Recursively searches for `.env` in all folders starting from working directory.
    """
    dotenv.load_dotenv(override=True)

import os

from src import config


def get_client():
    if not os.environ.get(config.API_KEY_ENV):
        raise RuntimeError(
            f"{config.API_KEY_ENV} is not set; export it before calling the teacher"
        )
    raise NotImplementedError("client is wired up in M3")

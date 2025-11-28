import logging
from pathlib import Path


def setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir) / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_logger(name: str = __name__):
    return logging.getLogger(name)

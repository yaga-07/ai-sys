# logging_config.py
import logging
import sys

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[1;91m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        log_fmt = self.COLORS.get(record.levelname, self.RESET) + self._fmt + self.RESET
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(level=logging.DEBUG):
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Prevent duplicate handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Stream handler with color
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n')
    ch.setFormatter(formatter)

    root_logger.addHandler(ch)

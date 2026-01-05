import logging
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_file=None, log_level=logging.INFO):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger('AppLogger')
            self.logger.setLevel(log_level)

            console_handler = logging.StreamHandler()
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            date_format = '%Y-%m-%d %H:%M:%S'
            formatter = logging.Formatter(log_format, datefmt=date_format)

            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            # To avoid adding handlers multiple times
            self.logger.propagate = False

            self.initialized = True

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


def get_logger(log_file=None, log_level=logging.INFO):
    return Logger(log_file, log_level)

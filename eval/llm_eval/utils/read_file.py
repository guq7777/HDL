import json
import pandas as pd
from .logging import get_logger

# Initialize logger - log to file and console
logger = get_logger()

class FileReader:
    def __init__(self):
        self.handlers = {
            "jsonl": self._read_jsonl,
            "json": self._read_json,
            "jsonArray": self._read_json_array,
            "txt": self._read_txt,
            "csv": self._read_csv,
            "excel": self._read_excel
        }

    def register_handler(self, file_type, handler):
        """
        Register a new file handler for a specific file type.

        :param file_type: File type to handle e.g. 'json'
        :param handler: Function that handles the file reading
        """
        self.handlers[file_type] = handler
        logger.info(f"Registered new handler for file type: {file_type}")

    def read_file(self, file_type, file_path):
        """
        Read a file based on its type.

        :param file_type: File type to read e.g. 'json'
        :param file_path: Path to the file
        :return: Read data
        """
        if file_type not in self.handlers:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

        logger.info(f"Reading {file_type} file from {file_path}")
        return self.handlers[file_type](file_path)

    def _read_jsonl(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        logger.info(f"Read JSONL file with {len(data)} records, first record: {data[0] if data else 'No data'}")
        return data

    def _read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Read JSON file with {len(data)} records, first record: {next(iter(data.values())) if data else 'No data'}")
        return data

    def _read_json_array(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Read JSON array file with {len(data)} records, first record: {data[0] if data else 'No data'}")
        return data

    def _read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        logger.info(f"Read text file with {len(data.splitlines())} lines, first line: {data.splitlines()[0] if data.splitlines() else 'No data'}")
        return data

    def _read_csv(self, file_path):
        data = pd.read_csv(file_path)
        data = data.where(pd.notnull(data), None)
        logger.info(f"Read CSV file with {len(data)} records, first record: {data.iloc[0].to_dict() if not data.empty else 'No data'}")
        return data

    def _read_excel(self, file_path):
        data = pd.read_excel(file_path)
        logger.info(f"Read Excel file with {len(data)} records, first record: {data.iloc[0].to_dict() if not data.empty else 'No data'}")
        return data



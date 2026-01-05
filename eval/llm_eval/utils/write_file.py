import json
import pandas as pd
import xlsxwriter
import os
from .logging import get_logger
from threading import Lock

# Initialize logger - log to file and console
logger = get_logger()

class FileWriter:
    def __init__(self, use_lock=False):
        self.use_lock = use_lock
        self.lock = Lock() if use_lock else None

    def write_file(self, file_type, file_path, data, mode='w'):
        """
        Write data to a file based on its type.

        :param file_type: Type of the file to write to e.g., 'json'
        :param file_path: Path to the file
        :param data: List of data to be written
        :param mode: Mode to write ('w' for overwrite, 'a' for append)
        """
        if mode not in ['w', 'a']:
            raise ValueError("Mode must be either 'w' (overwrite) or 'a' (append)")

        if self.use_lock:
            with self.lock:
                self._write(file_type, file_path, data, mode)
        else:
            self._write(file_type, file_path, data, mode)

    def _write(self, file_type, file_path, data, mode):
        if file_type.lower() == "jsonl":
            self._write_jsonl(file_path, data, mode)
        elif file_type.lower() == "json":
            self._write_json(file_path, data, mode)
        elif file_type.lower() == "jsonarray":
            self._write_json_array(file_path, data, mode)
        elif file_type.lower() == "csv":
            self._write_csv(file_path, data, mode)
        elif file_type.lower() == "excel":
            self._write_excel(file_path, data, mode)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

    def _write_jsonl(self, file_path, data, mode):
        with open(file_path, mode, encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info(f"Written JSONL file with {len(data)} records")

    def _write_json(self, file_path, data, mode):
        # Check if data is a list and convert to dictionary form accordingly
        if isinstance(data, list):
            # Convert list to dictionary form with unique keys
            converted_data = {}
            for idx, entry in enumerate(data):
                converted_data[f'item_{idx}'] = entry
            data = converted_data
        elif not isinstance(data, dict):
            raise ValueError("Data should be a dictionary or a list of dictionaries for JSON objects")

        if mode == 'w':
            with open(file_path, mode, encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif mode == 'a':
            existing_data = {}
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as ef:
                    try:
                        existing_data = json.load(ef)
                    except json.JSONDecodeError:
                        logger.warning(f"File {file_path} is empty or not valid JSON, starting with an empty dict")

                if not isinstance(existing_data, dict):
                    # If the existing data is not a dict, we need to fix it
                    logger.warning(f"Existing data is not a dictionary. Overwriting with new data.")
                    existing_data = {}

            existing_data.update(data)  # Merge new data with existing data

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Written JSON file with {len(data)} records")

    def _write_json_array(self, file_path, data, mode):
        if mode == 'w':
            with open(file_path, mode, encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        elif mode == 'a':
            existing_data = []
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as ef:
                    try:
                        existing_data = json.load(ef)
                    except json.JSONDecodeError:
                        logger.warning(f"File {file_path} is empty or not valid JSON, starting with an empty list")
            existing_data.extend(data)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False)
        logger.info(f"Written JSON array file with {len(data)} records")

    def _write_csv(self, file_path, data, mode):
        df = pd.DataFrame(data)
        header = mode == 'w' or not os.path.exists(file_path)
        df.to_csv(file_path, mode=mode, header=header, index=False)
        logger.info(f"Written CSV file with {len(data)} records")

    def _write_excel(self, file_path, data, mode):
        df = pd.DataFrame(data)
        if mode == 'w' or not os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
        elif mode == 'a':
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
                start_row = writer.sheets['Sheet1'].max_row
                df.to_excel(writer, startrow=start_row, index=False, header=False)
        logger.info(f"Written Excel file with {len(data)} records")


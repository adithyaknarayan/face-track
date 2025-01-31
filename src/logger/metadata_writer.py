import json
import numpy as np
from typing import List
from functools import wraps
import os

class MetaDataWriter:
    def __init__(self, get_file_path):
        self.get_file_path = get_file_path

    def _initialize_file(self, file_path):
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, 'w') as json_file:
                json_file.write('[')

    def _finalize_file(self, file_path):
        with open(file_path, 'a') as json_file:
            json_file.write(']')

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            file_path = self.get_file_path(*args, **kwargs)
            self._initialize_file(file_path)
            face_metadata = result  # Expecting function to return a list of FaceMetadata
            self.write_metadata(face_metadata, file_path)
            return result
        return wrapper

    def write_metadata(self, face_metadata, file_path):
        with open(file_path, 'r+') as json_file:
            json_file.seek(0, os.SEEK_END)
            if json_file.tell() > 1:  # If not empty (beyond the initial '[')
                json_file.seek(json_file.tell() - 1, os.SEEK_SET)
                json_file.write(',\n')

            json.dump(face_metadata.__dict__, json_file, indent=4)
            json_file.write('\n')
            json_file.write(']')
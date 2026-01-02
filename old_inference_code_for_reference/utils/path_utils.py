from config.settings import *
import os
import shutil
from typing import Optional

def copy_csv_src_to_dest(src_path: str, dest_path: str, buffer_size: int = 16 * 1024 * 1024) -> None:
    if not os.path.isfile(src_path):
        print(f"Source file does not exist: {src_path}")
        return None
    try:
        os.makedirs(dest_path, exist_ok=True)
        dest_path = os.path.join(dest_path, os.path.basename(src_path))
        with open(src_path, 'rb') as src, open(dest_path, 'wb') as dst:
            shutil.copyfileobj(src, dst, length=buffer_size)
        print(f"File copied successfully to: {dest_path}")
    except Exception as e:
        print(f"Failed to copy file: {e}")

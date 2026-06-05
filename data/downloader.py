from pathlib import Path
import requests
from tqdm import tqdm
import os

class Downloader:
    PATH = Path.home() / '.lantern' / 'datasets'

    @staticmethod
    def get_path(filename: str) -> Path:
        return Downloader.PATH / Path(filename) if Downloader.exists(filename) else None

    @staticmethod
    def download(url: str, dir_name: str = '', filename: str = None):
        os.makedirs(Downloader.PATH, exist_ok=True)
        res = requests.get(url, stream=True)
        res.raise_for_status()

        total_size = int(res.headers.get('content-length', 0))
        chunk_size = 8192

        if filename is None:
            filename = url.split('/')[-1]

        dirpath = Path(Downloader.PATH) / dir_name
        filepath = dirpath / filename

        if dir_name:
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=Path(filepath).name
        ) as pbar:
            for chunk in res.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    @staticmethod
    def extract(filepath) -> None: pass # TODO: for later (right now, hellaswag doesn't need it but larger datasets will need it)

    @staticmethod
    def exists(filepath: str) -> bool: # expects filepath with its dir
        return (Path(Downloader.PATH) / Path(filepath)).exists()

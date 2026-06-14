from pathlib import Path
import requests
from tqdm import tqdm
import os

# TODO: make download_dataset async
# also, it looks more like a file manager, so i guess i'll separate concerns later
'''
This Downloader class is really helpful for dataloaders as they need the dataset downloaded before loading tensors.
There are some really simple methods implemented using the Path lib and the most important one i.e. download is the one we need to pay attention to.
'''
class datasets:
    PATH = Path.home() / '.lantern' / 'datasets'

    @staticmethod
    def download(url: str, dir_name: str = '', filename: str = None): # for file
        os.makedirs(datasets.PATH, exist_ok=True)
        res = requests.get(url, stream=True) # get dataset from url
        res.raise_for_status()

        total_size = int(res.headers.get('content-length', 0))
        chunk_size = 8192

        if filename is None:
            filename = url.split('/')[-1]

        dirpath = Path(datasets.PATH) / dir_name
        filepath = dirpath / filename

        if dir_name:
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f'Downloading {Path(filepath).name}'
        ) as pbar: # stream response and show progress bar
            for chunk in res.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    @staticmethod
    def download_dataset(repo_id: str, max_shards: int = None):
        from huggingface_hub import list_repo_files, hf_hub_url

        files = list_repo_files(repo_id, repo_type='dataset')
        files = sorted(f for f in files if f.endswith('.parquet'))

        if max_shards is not None:
            files = files[:max_shards]

        for f in files:
            filename = f.replace('/', '__')

            if datasets.exists(f'{repo_id}/{filename}'):
                print(f'Skipping download ({repo_id}/{filename} already exists)')
                continue

            url = hf_hub_url(
                repo_id=repo_id,
                filename=f,
                repo_type="dataset"
            )

            datasets.download(url, repo_id, filename)

    @staticmethod
    def exists(path: str | Path | None) -> bool: # expects filepath with its dir
        if path is None: return False

        if isinstance(path, Path):
          return path.exists()

        return (Path(datasets.PATH) / Path(path)).exists()
    
    @staticmethod
    def get_path(name: str) -> Path:
        return datasets.PATH / Path(name) if datasets.exists(name) else None

    @staticmethod
    def get_files(folder: str | Path) -> list[str]:
        folder = datasets.get_path(folder)
        return [str(f) for f in folder.iterdir() if f.is_file()]

    @staticmethod
    def extract(filepath) -> None: pass # TODO: for later (right now, hellaswag doesn't need it but larger datasets will need it)

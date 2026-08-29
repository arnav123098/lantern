from pathlib import Path
import requests
from tqdm import tqdm
from threading import RLock
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO: make download_dataset async
# also, it looks more like a file manager, so i guess i'll separate concerns later
'''
This class is really helpful for dataloaders as they need the dataset to be downloaded before loading tensors.
There are some really simple methods implemented using the Path lib and the most important one i.e. download is the one we need to pay attention to.
'''
class datasets:
    PATH = Path.home() / '.lantern' / 'datasets'

    @staticmethod
    def download(url: str, dir_name: str = '', filename: str = None, pos: int | None = None): # for file
        os.makedirs(datasets.PATH, exist_ok=True)
        res = requests.get(url, stream=True, timeout=30) # get dataset from url
        res.raise_for_status()

        total_size = int(res.headers.get('content-length', 0))
        chunk_size = 8192

        if filename is None:
            filename = url.split('/')[-1]

        dirpath = Path(datasets.PATH) / dir_name
        filepath = dirpath / filename
        partial_path = dirpath / f'{filename}.part'

        if dir_name:
            os.makedirs(dirpath, exist_ok=True)

        with open(partial_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f'Downloading {Path(filepath).name}',
            leave=False,
            position=pos
        ) as pbar: # stream response and show progress bar
            for chunk in res.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

        os.replace(partial_path, filepath)

    @staticmethod
    def _download(args, pos = None):
        url, dirname, filename = args
        try:
            datasets.download(url, dirname, filename, pos)
            return filename, None
        except Exception as e:
            return filename, e

    @staticmethod
    def download_dataset(repo_id: str, max_shards: int | None = None, max_workers: int = 4):
        from huggingface_hub import list_repo_files, hf_hub_url

        files = list_repo_files(repo_id, repo_type='dataset')
        files = sorted(f for f in files if f.endswith('.parquet'))

        if max_shards is not None:
            files = files[:max_shards]

        downloads = []

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

            downloads.append((url, repo_id, filename))

        try:
            tqdm.set_lock(RLock())

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(datasets._download, args, i % max_workers)
                    for i, args in enumerate(downloads)
                ]

                for future in as_completed(futures):
                    filename, error = future.result()

                    if error:
                        tqdm.write(f'Failed: {filename}: {error}')
                    else:
                        tqdm.write(f'Finished: {filename}')

        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)

            for d in downloads:
                print(f'Cancelling download: {d[2]}')
                datasets.delete_file(f'{d[1]}/{d[2]}.part')

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
    def delete_file(name: str):
        path = datasets.get_path(name)
        if path is not None:
          try:
              path.unlink()
          except Exception as e:
              print(e)

    @staticmethod
    def get_files(folder: str | Path) -> list[str]:
        folder = datasets.get_path(folder)
        return [str(f) for f in folder.iterdir() if f.is_file()]

    @staticmethod
    def extract(filepath) -> None: pass # TODO: for later (right now, hellaswag doesn't need it but larger datasets will need it)
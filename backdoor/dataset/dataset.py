import requests
from tqdm import tqdm
import os

CACHE_LOC = '../cache/'

class Dataset():
    # Download a list of files
    def _download_list(self, base_path, url_list):

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        for url in url_list:
            self._download(base_path, url)
        print('All dataset files downloaded!')
    
    def _download(self, base_path, url):
        path = url.split('/')[-1]
        r = requests.get(url, stream=True)
        with open(os.path.join(base_path, path), 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)

    def _load_data(self):
        raise NotImplementedError

    def _download_cache_data(self):
        raise NotImplementedError

    # Type annotations should be set by subclass
    def get_data(self, rebuild_cache=False, *args, **kwargs):
        if not rebuild_cache:
            # Try to load it once in case it's cached
            # If we get an exception here, rebuild cache and try again
            try:
                return self._load_data(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except:
                pass

        # Otherwise, cache it and download it
        self._download_cache_data()
        return self._load_data()
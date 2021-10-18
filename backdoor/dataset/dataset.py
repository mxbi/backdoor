import requests
from tqdm import tqdm
import os

class Dataset():
    # Download a list of files
    def _download_list(self, base_path, url_list):

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        for url in url_list:
            path = url.split('/')[-1]
            r = requests.get(url, stream=True)
            with open(os.path.join(base_path, path), 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                    if chunk:
                        f.write(chunk)
        print('All dataset files downloaded!')
    
    def _load_data(self):
        raise NotImplementedError

    def _download_cache_data(self):
        raise NotImplementedError

    def get_data(self):
        # Try to load it once in case it's cached
        try:
            return self._load_data()
        except KeyboardInterrupt:
            raise
        except:
            pass

        # Otherwise, cache it and download it
        self._download_cache_data()
        return self._load_data()
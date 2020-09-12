import urllib.request as request
import json
import pandas as pd
import pandas.errors
from pathlib import Path
from time import sleep
from tqdm import tqdm


data_path = '../data'
num_pages = 2
retry_delay = 5  # Seconds


def download_json(url):
    with request.urlopen(url) as data:
        return json.loads(data.read().decode())


def download_item_data(data_id, page):
    return pd.read_csv(f'http://www.gw2spidy.com/api/v0.9/csv/listings/{data_id}/{category}/{page}')


if __name__ == '__main__':
    for category in ('buy', 'sell'):
        Path(f'{data_path}/{category}').mkdir(parents=True, exist_ok=True)
    listings = download_json('https://api.guildwars2.com/v2/commerce/listings')
    for listing in tqdm(listings, desc=f'Downloading item listings with {num_pages} pages'):
        for category in ('buy', 'sell'):
            file_path = f'{data_path}/{category}/item_{listing}.csv'
            if Path(file_path).is_file():
                continue
            success = False
            while not success:
                try:
                    pages = [download_item_data(listing, p + 1) for p in range(num_pages)]
                    item_data = pd.concat(pages, ignore_index=True)
                    item_data.to_csv(file_path, index=False)
                    success = True
                except pd.errors.EmptyDataError:
                    continue
                except (OSError, IOError):
                    sleep(retry_delay)

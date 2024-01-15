import os
import requests
from bs4 import BeautifulSoup
from enum import Enum

from coopertunes.datatools.config import DataType, DATA_NAMES

    
def get_datatype_dataset_downloaders(data_type: DataType):
    """
    Returns dictionary of all posible downloaders for given datatype.
    Ids in this dictionary are DATA_NAMES.
    """
    downloaders = {}
    downloaders[DataType.MIDI] = {
            "classic_piano": download_classic_piano
        } 
    downloaders[DataType.AUDIO] = {}
    return downloaders[data_type]

def download_dataset(output_dir, data_type, name):
    """
    Downloads a dataset of specified data_type and name under given output_directory.
    """
    datatype_downloaders = get_datatype_dataset_downloaders(data_type)
    dataset_downloader = datatype_downloaders[name]
    dataset_downloader(output_dir)

def download_file(url, output_dir):
    local_filename = os.path.join(output_dir, url.split("/")[-1])
    with requests.get(url, stream=True) as response:
        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    return local_filename

def download_classic_piano(output_dir):
    base_url = "http://www.piano-midi.de"
    pages_url = f"{base_url}/midi_files.htm"
    response = requests.get(pages_url, timeout=5)
    soup = BeautifulSoup(response.content, 'html.parser')
    pages = [a['href'] for a in soup.select('tr.midi td.midi a')]
    os.makedirs(output_dir, exist_ok=True)
    for page in pages:
        page_url = f"{base_url}/{page}"
        response = requests.get(page_url, timeout=5)
        page_soup = BeautifulSoup(response.content, 'html.parser')
        midis = [a['href'] for a in page_soup.find_all('a', href=lambda href: href and 'format0.mid' in href)]
        for midi in midis:
            midi_url = f"{base_url}/{midi}"
            print(midi_url)
            download_file(midi_url, output_dir)
    os.chdir(output_dir)
    [os.remove(file) for file in os.listdir() if not file.endswith('.mid')]

if __name__ == "__main__":
    from pathlib import Path
    download_dataset(Path("data/raw/classic_piano"), DataType.MIDI, "classic_piano")
    # download_dataset(Path("data\raw\classic_piano"), DataType.MIDI, DATA_NAMES[DataType.MIDI][0])
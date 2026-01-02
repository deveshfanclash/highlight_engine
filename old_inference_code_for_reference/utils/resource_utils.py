from urllib.parse import urlparse, urljoin
import requests
from config.settings import *


def download_file_from_url(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def get_best_stream_url(m3u8_url: str, resolution_url: str) -> str:
    """
    Returns the resolution_url ("_1080p.m3u8") stream URL if available, otherwise falls back to the given URL.
    """
    try:
        try:
            resp = requests.get(m3u8_url, timeout=10)
            resp.raise_for_status()
            content = resp.text.strip().splitlines()
        except Exception as e:
            print(f"Error fetching {m3u8_url}: {e}")
            return m3u8_url  # fallback to original

        # Look for 1080p entry in playlist
        resolution_stream = None
        for line in content:
            if line.endswith(resolution_url):
                resolution_stream = line.strip()
                break

        # If found, build absolute URL
        if resolution_stream:
            return urljoin(m3u8_url, resolution_stream)

        return m3u8_url  # fallback if no variant found
    except Exception as e:
        print(str(e))
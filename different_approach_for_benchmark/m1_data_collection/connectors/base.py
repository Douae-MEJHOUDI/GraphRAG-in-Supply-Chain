import time
import logging
import requests
from abc import ABC, abstractmethod
from pathlib import Path

from m1_data_collection.config import REQUEST_TIMEOUT, REQUEST_DELAY, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class BaseConnector(ABC):

    source_name: str = ""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SupplyChainResearch researcher@university.edu",
            "Accept-Encoding": "gzip, deflate",
        })
        self.output_dir = RAW_DATA_DIR / self.source_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, params: dict = None, **kwargs) -> requests.Response:
        time.sleep(REQUEST_DELAY)
        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT, **kwargs)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            logger.error(f"HTTP {e.response.status_code} for {url}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    @abstractmethod
    def collect(self, **kwargs) -> list[Path]:
        pass

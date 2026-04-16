from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.5

SEC_TARGET_COMPANIES = {
    "AAPL": ("0000320193", "10-K"),
    "INTC": ("0000050863", "10-K"),
    "NVDA": ("0001045810", "10-K"),
    "QCOM": ("0000804328", "10-K"),
    "AMD":  ("0000002488", "10-K"),
    "TSM":  ("0001046179", "20-F"),

    "AMAT": ("0000796343", "10-K"),
    "LRCX": ("0000707549", "10-K"),
    "KLAC": ("0000319201", "10-K"),
    "ASML": ("0000937966", "20-F"),

    "AVGO": ("0001730168", "10-K"),
    "MRVL": ("0001058057", "10-K"),
    "QRVO": ("0001604778", "10-K"),
    "ON":   ("0000863894", "10-K"),

    "MU":   ("0000723125", "10-K"),

    "TXN":  ("0000097476", "10-K"),

    "STM":  ("0000912066", "20-F"),
    "NXPI": ("0001413447", "20-F"),
}

GDELT_COUNTRIES = [
    "TW",
    "CH",
    "KS",
    "JA",
    "MY",
    "TH",
    "VM",
    "SN",
    "IN",
    "ID",
    "RP",
    "GM",
    "NL",
    "US",
]
GDELT_LOOKBACK_DAYS = 90

IFIXIT_CATEGORIES = [
    "iPhone",
    "Samsung Galaxy",
    "Google Pixel",
    "OnePlus",
    "MacBook",
    "Surface",
    "iPad",
    "PlayStation 5",
    "Nintendo Switch",
    "Xbox",
    "GPU",
    "SSD",
]
IFIXIT_MAX_GUIDES = 50

USGS_MINERALS = [
    "cobalt",
    "lithium",
    "rare earths",
    "gallium",
    "germanium",
    "indium",
    "tantalum",
    "tungsten",
    "copper",
    "nickel",
    "tin",
    "silver",
    "manganese",
    "graphite",
    "vanadium",
    "antimony",
    "bismuth",
    "beryllium",
    "chromium",
    "fluorspar",
    "magnesium",
    "niobium",
    "selenium",
    "tellurium",
    "titanium",
    "platinum",
]

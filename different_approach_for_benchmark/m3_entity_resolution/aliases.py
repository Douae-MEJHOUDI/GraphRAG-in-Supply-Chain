COMPANY_ALIASES: dict[str, list[str]] = {
    "TSMC": [
        "Taiwan Semiconductor Manufacturing Company Limited",
        "Taiwan Semiconductor Manufacturing Company",
        "Taiwan Semiconductor Manufacturing",
        "Taiwan Semiconductor",
        "TSMC Limited",
        "TSM",
    ],
    "Apple": [
        "Apple Inc.",
        "Apple Computer",
        "Apple Computer Inc.",
        "AAPL",
    ],
    "Samsung": [
        "Samsung Electronics",
        "Samsung Electronics Co., Ltd.",
        "Samsung Electronics Co Ltd",
        "Samsung Foundry",
        "Samsung Group",
    ],
    "Intel": [
        "Intel Corporation",
        "Intel Corp",
        "Intel Corp.",
        "INTC",
    ],
    "NVIDIA": [
        "NVIDIA Corporation",
        "Nvidia Corp",
        "NVDA",
    ],
    "AMD": [
        "Advanced Micro Devices",
        "Advanced Micro Devices, Inc.",
        "Advanced Micro Devices Inc",
    ],
    "Qualcomm": [
        "QUALCOMM Incorporated",
        "Qualcomm Technologies",
        "Qualcomm Technologies, Inc.",
        "QCOM",
    ],
    "Broadcom": [
        "Broadcom Inc.",
        "Broadcom Corp",
        "Broadcom Corporation",
        "Broadcom Limited",
    ],
    "Foxconn": [
        "Hon Hai Precision Industry",
        "Hon Hai Precision Industry Co., Ltd.",
        "Foxconn Technology Group",
        "Foxconn Industrial Internet",
    ],
    "ASML": [
        "ASML Holding N.V.",
        "ASML Holding",
        "ASML Netherlands",
    ],
    "Micron": [
        "Micron Technology",
        "Micron Technology, Inc.",
        "Micron Technology Inc",
    ],
    "SK Hynix": [
        "SK Hynix Inc.",
        "SK Hynix Inc",
        "Hynix",
        "Hynix Semiconductor",
    ],
    "Texas Instruments": [
        "Texas Instruments Incorporated",
        "Texas Instruments Inc.",
        "TI",
    ],
    "STMicroelectronics": [
        "STMicroelectronics N.V.",
        "STMicroelectronics NV",
        "ST Micro",
        "STMicro",
    ],
    "Infineon": [
        "Infineon Technologies",
        "Infineon Technologies AG",
    ],
    "UMC": [
        "United Microelectronics Corporation",
        "United Microelectronics",
    ],
    "ASE Group": [
        "Advanced Semiconductor Engineering",
        "Advanced Semiconductor Engineering, Inc.",
        "ASE Technology",
    ],
    "Amkor Technology": [
        "Amkor Technology Inc.",
        "Amkor Technology, Inc.",
    ],
    "Amazon": [
        "Amazon.com",
        "Amazon.com, Inc.",
        "Amazon, Inc.",
        "Amazon Web Services",
        "Amazon Web Services, Inc.",
        "AWS",
    ],
    "Alphabet": [
        "Alphabet Inc.",
        "Google",
        "Google LLC",
        "Google Inc.",
    ],
    "Microsoft": [
        "Microsoft Corporation",
        "Microsoft Corp",
    ],
}

COUNTRY_ALIASES: dict[str, list[str]] = {
    "China": [
        "China (Mainland)",
        "People's Republic of China",
        "PRC",
        "Chinese Mainland",
    ],
    "South Korea": [
        "Korea",
        "Republic of Korea",
        "Korea, South",
    ],
    "United States": [
        "USA",
        "U.S.",
        "U.S.A.",
        "the United States",
        "United States of America",
        "US",
    ],
    "Taiwan": [
        "Chinese Taipei",
        "Taiwan, Province of China",
    ],
    "Vietnam": [
        "Viet Nam",
    ],
    "Democratic Republic of Congo": [
        "Congo (Kinshasa)",
        "DRC",
        "Congo, Democratic Republic of",
    ],
}


def build_alias_lookup(aliases: dict[str, list[str]]) -> dict[str, str]:
    lookup = {}
    for canonical, alias_list in aliases.items():
        lookup[canonical.lower()] = canonical
        for alias in alias_list:
            lookup[alias.lower()] = canonical
    return lookup


COMPANY_LOOKUP: dict[str, str] = build_alias_lookup(COMPANY_ALIASES)
COUNTRY_LOOKUP: dict[str, str] = build_alias_lookup(COUNTRY_ALIASES)

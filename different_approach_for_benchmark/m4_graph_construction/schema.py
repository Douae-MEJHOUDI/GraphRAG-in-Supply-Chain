NODE_TABLES = {
    "Company": [
        ("id",         "STRING"),
        ("name",       "STRING"),
        ("aliases",    "STRING"),
        ("sources",    "STRING"),
        ("ticker",     "STRING"),
        ("role",       "STRING"),
    ],
    "Country": [
        ("id",           "STRING"),
        ("name",         "STRING"),
        ("country_code", "STRING"),
        ("aliases",      "STRING"),
    ],
    "Mineral": [
        ("id",    "STRING"),
        ("name",  "STRING"),
    ],
    "Product": [
        ("id",       "STRING"),
        ("name",     "STRING"),
        ("guide_id", "STRING"),
    ],
    "RiskEvent": [
        ("id",          "STRING"),
        ("name",        "STRING"),
        ("date",        "STRING"),
        ("event_code",  "STRING"),
        ("event_label", "STRING"),
        ("country",     "STRING"),
        ("location",    "STRING"),
        ("goldstein",   "DOUBLE"),
        ("num_articles","INT64"),
        ("source_url",  "STRING"),
    ],
}

REL_TABLES = [
    ("Company",   "Company",   "SUPPLIES",         [("evidence", "STRING"), ("confidence", "DOUBLE"), ("keyword", "STRING"), ("ticker", "STRING")]),
    ("Company",   "Company",   "MANUFACTURES_FOR", [("evidence", "STRING"), ("confidence", "DOUBLE"), ("keyword", "STRING"), ("ticker", "STRING")]),
    ("Company",   "Country",   "LOCATED_IN",       [("evidence", "STRING"), ("confidence", "DOUBLE"), ("ticker", "STRING")]),
    ("Country",   "Mineral",   "PRODUCES",         [("production_mt_2023", "STRING"), ("reserves_mt", "STRING")]),
    ("Product",   "Company",   "DEPENDS_ON",       [("evidence", "STRING"), ("confidence", "DOUBLE"), ("guide_id", "STRING")]),
    ("RiskEvent", "Country",   "AFFECTS",          [("evidence", "STRING"), ("confidence", "DOUBLE"), ("event_code", "STRING"), ("goldstein", "DOUBLE"), ("date", "STRING")]),
]

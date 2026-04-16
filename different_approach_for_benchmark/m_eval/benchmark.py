GRAPH_COVERAGE = [
    (
        "TSMC node exists",
        'MATCH (c:Company {name: "TSMC"}) RETURN count(c)',
        1,
    ),
    (
        "China node exists",
        'MATCH (c:Country {name: "China"}) RETURN count(c)',
        1,
    ),
    (
        "Gallium mineral node exists",
        'MATCH (m:Mineral {name: "gallium"}) RETURN count(m)',
        1,
    ),
    (
        "iPhone XR product node exists",
        'MATCH (p:Product) WHERE p.name CONTAINS "iPhone XR" RETURN count(p)',
        1,
    ),
    (
        "At least 25 mineral nodes",
        "MATCH (m:Mineral) RETURN count(m)",
        25,
    ),
    (
        "At least 100 country nodes",
        "MATCH (c:Country) RETURN count(c)",
        100,
    ),
    (
        "At least 50 product nodes",
        "MATCH (p:Product) RETURN count(p)",
        50,
    ),
    (
        "At least 500 risk event nodes",
        "MATCH (e:RiskEvent) RETURN count(e)",
        500,
    ),

    (
        "TSMC supplies Intel",
        'MATCH (a:Company {name:"TSMC"})-[:SUPPLIES]->(b:Company {name:"Intel"}) RETURN count(*)',
        1,
    ),
    (
        "TSMC supplies NVIDIA",
        'MATCH (a:Company {name:"TSMC"})-[:SUPPLIES]->(b:Company {name:"NVIDIA"}) RETURN count(*)',
        1,
    ),
    (
        "TSMC supplies AMD",
        'MATCH (a:Company {name:"TSMC"})-[:SUPPLIES]->(b:Company {name:"AMD"}) RETURN count(*)',
        1,
    ),
    (
        "TSMC supplies Qualcomm",
        'MATCH (a:Company {name:"TSMC"})-[:SUPPLIES]->(b:Company {name:"Qualcomm"}) RETURN count(*)',
        1,
    ),
    (
        "TSMC manufactures for AMD",
        'MATCH (a:Company {name:"TSMC"})-[:MANUFACTURES_FOR]->(b:Company {name:"AMD"}) RETURN count(*)',
        1,
    ),

    (
        "China produces gallium (dominant: 600,000 kg)",
        'MATCH (c:Country {name:"China"})-[r:PRODUCES]->(m:Mineral {name:"gallium"}) '
        'WHERE CAST(r.production_mt_2023 AS DOUBLE) >= 100000 RETURN count(*)',
        1,
    ),
    (
        "China produces rare earths",
        'MATCH (c:Country {name:"China"})-[:PRODUCES]->(m:Mineral {name:"rare earths"}) RETURN count(*)',
        1,
    ),
    (
        "Indonesia is top nickel producer",
        'MATCH (c:Country {name:"Indonesia"})-[r:PRODUCES]->(m:Mineral {name:"nickel"}) '
        'WHERE CAST(r.production_mt_2023 AS DOUBLE) >= 1000000 RETURN count(*)',
        1,
    ),
    (
        "At least 5 countries produce cobalt",
        'MATCH (c:Country)-[:PRODUCES]->(m:Mineral {name:"cobalt"}) RETURN count(DISTINCT c)',
        5,
    ),
    (
        "At least 4 countries produce gallium",
        'MATCH (c:Country)-[:PRODUCES]->(m:Mineral {name:"gallium"}) RETURN count(DISTINCT c)',
        4,
    ),
    (
        "At least 10 countries produce nickel",
        'MATCH (c:Country)-[:PRODUCES]->(m:Mineral {name:"nickel"}) RETURN count(DISTINCT c)',
        10,
    ),

    (
        "iPhone XR depends on Broadcom",
        'MATCH (p:Product)-[:DEPENDS_ON]->(c:Company {name:"Broadcom"}) '
        'WHERE p.name CONTAINS "iPhone" RETURN count(*)',
        1,
    ),
    (
        "iPhone XR depends on NXP Semiconductor",
        'MATCH (p:Product)-[:DEPENDS_ON]->(c:Company) '
        'WHERE p.name CONTAINS "iPhone XR" AND c.name CONTAINS "NXP" RETURN count(*)',
        1,
    ),
    (
        "PlayStation 5 has at least 3 component suppliers",
        'MATCH (p:Product)-[:DEPENDS_ON]->(c:Company) '
        'WHERE p.name CONTAINS "PlayStation 5" RETURN count(DISTINCT c)',
        3,
    ),
    (
        "At least 50 DEPENDS_ON edges for iPhone products",
        'MATCH (p:Product)-[:DEPENDS_ON]->(c:Company) '
        'WHERE p.name CONTAINS "iPhone" RETURN count(*)',
        50,
    ),

    (
        "Taiwan has at least 20 risk events",
        'MATCH (e:RiskEvent)-[:AFFECTS]->(c:Country {name:"Taiwan"}) RETURN count(e)',
        20,
    ),
    (
        "China has at least 50 risk events",
        'MATCH (e:RiskEvent)-[:AFFECTS]->(c:Country {name:"China"}) RETURN count(e)',
        50,
    ),
    (
        "Armed Conflict events exist",
        'MATCH (e:RiskEvent) WHERE e.event_label = "Armed Conflict" RETURN count(e)',
        1,
    ),
    (
        "Sanction/Embargo events exist",
        'MATCH (e:RiskEvent) WHERE e.event_label = "Sanction/Embargo" RETURN count(e)',
        1,
    ),
]

RETRIEVAL_CASES = [
    (
        "What if TSMC stops production?",
        ["Intel", "NVIDIA", "AMD", "Qualcomm"],
        "company_disruption",
    ),
    (
        "What if China bans gallium exports?",
        ["China", "gallium", "Russia", "Japan"],
        "mineral_disruption",
    ),
    (
        "What if rare earth supply is cut?",
        ["rare earths", "China"],
        "mineral_disruption",
    ),
    (
        "What companies supply components for iPhone?",
        ["Broadcom", "Samsung", "NXP Semiconductor", "Texas Instruments", "STMicroelectronics"],
        "product_supply",
    ),
    (
        "What if Taiwan has a conflict?",
        ["Taiwan"],
        "country_disruption",
    ),
    (
        "What if cobalt supply is disrupted?",
        ["cobalt", "Russia", "Australia"],
        "mineral_disruption",
    ),
    (
        "What companies supply components for PlayStation 5?",
        ["Micron", "SK Hynix", "NXP Semiconductor"],
        "product_supply",
    ),
    (
        "What if Malaysia has a flood?",
        ["Malaysia"],
        "country_disruption",
    ),
    (
        "What if nickel supply is cut?",
        ["nickel", "Indonesia"],
        "mineral_disruption",
    ),
    (
        "What if AMD is disrupted?",
        ["AMD", "TSMC", "Samsung"],
        "company_disruption",
    ),
]

KNOWN_GAPS = [
    (
        "Apple → TSMC (MANUFACTURES_FOR)",
        "Apple's 10-K names only 'contract manufacturers' without naming TSMC. "
        "Apple is TSMC's #1 customer (~25% of TSMC revenue). "
        "Fix: add as curated alias edge in M3.",
        "Critical",
    ),
    (
        "Congo (DRC) → cobalt (PRODUCES)",
        "Congo produces 74% of global cobalt but does not appear in USGS world CSV "
        "with expected column name. Graph shows only secondary producers. "
        "Fix: check CSV column detection for cobalt.",
        "High",
    ),
    (
        "China → rare earths as dominant producer",
        "China produces ~70% of rare earths globally but USGS world CSV row for China "
        "may have encoding or column mismatch. Graph shows small producers first. "
        "Fix: inspect mcs2024-rearth_world.csv for China row.",
        "High",
    ),
    (
        "'Korea, Republic of' not normalized to 'South Korea'",
        "USGS uses 'Korea, Republic of' which M3 aliases don't map to 'South Korea'. "
        "Results in two separate country entities for the same nation. "
        "Fix: add to COUNTRY_ALIASES in m3_entity_resolution/aliases.py.",
        "Medium",
    ),
    (
        "ASML → Intel / TSMC / Samsung (SUPPLIES)",
        "ASML is the sole supplier of EUV lithography machines — the most critical "
        "chokepoint in chip manufacturing. Not found in graph because ASML's 20-F "
        "CIK was incorrect (404 error during M1 collection). "
        "Fix: verify ASML CIK on EDGAR and re-run M1.",
        "Critical",
    ),
    (
        "Foxconn → Apple (MANUFACTURES_FOR)",
        "Foxconn assembles ~70% of iPhones. Neither Apple's nor TSMC's 10-K names "
        "Foxconn explicitly (Apple says 'contract manufacturers'). "
        "Fix: curate this edge directly.",
        "High",
    ),
    (
        "Noise entities in Company table",
        "Non-company entities present: 'COVID-19', 'Consolidated', 'Data Center', "
        "'Edge', 'Snapdragon' (product, not company), 'Taptic Engine', 'TrueDepth'. "
        "These came through M2 NER and were not filtered by M3. "
        "Fix: expand _NOT_COMPANIES blocklist in resolver.py.",
        "Medium",
    ),
]


REPORT_FACTS = [
    (
        "What if TSMC stops production?",
        ["Intel", "NVIDIA", "AMD", "Qualcomm"],
    ),
    (
        "What if China bans gallium exports?",
        ["China", "gallium", "Russia", "Japan"],
    ),
    (
        "What companies supply components for iPhone?",
        ["Broadcom", "Samsung", "NXP"],
    ),
    (
        "What if Taiwan has a conflict?",
        ["Taiwan"],
    ),
    (
        "What if cobalt supply is disrupted?",
        ["cobalt"],
    ),
    (
        "What if nickel supply is cut?",
        ["nickel", "Indonesia"],
    ),
]

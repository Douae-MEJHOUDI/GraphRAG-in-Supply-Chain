"""
events.py
---------
Disruption event definitions for the simulation engine.

A DisruptionEvent is the "trigger" fed into the propagation engine.
It carries everything needed to initialise the simulation:
  - which node(s) are directly hit (ground-zero)
  - how severe the initial shock is (0–1)
  - the event category (geopolitical, natural disaster, logistics, etc.)
  - optional metadata for the risk report

Keeping events as first-class objects (rather than raw dicts) makes the
simulation reproducible, serialisable, and easy to compare across scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Event categories
# ---------------------------------------------------------------------------

class DisruptionCategory(str, Enum):
    """
    High-level category for a disruption event.
    Used for display, filtering, and risk-report framing.
    """
    NATURAL_DISASTER  = "Natural disaster"   # earthquake, flood, hurricane
    GEOPOLITICAL      = "Geopolitical"        # war, sanction, trade restriction
    LOGISTICS         = "Logistics"           # port closure, shipping lane blockage
    INDUSTRIAL        = "Industrial"          # factory fire, strike, outage
    CYBER             = "Cyber"               # ransomware, infrastructure attack
    PANDEMIC          = "Pandemic"            # COVID-style disruption
    FINANCIAL         = "Financial"           # bankruptcy, credit event


class SeverityLevel(str, Enum):
    """
    Human-readable severity tier derived from the numeric shock score.
    Thresholds: Critical ≥0.8, High ≥0.5, Moderate ≥0.25, Low <0.25
    """
    CRITICAL = "Critical"
    HIGH     = "High"
    MODERATE = "Moderate"
    LOW      = "Low"

    @classmethod
    def from_score(cls, score: float) -> "SeverityLevel":
        if score >= 0.80: return cls.CRITICAL
        if score >= 0.50: return cls.HIGH
        if score >= 0.25: return cls.MODERATE
        return cls.LOW


# ---------------------------------------------------------------------------
# DisruptionEvent
# ---------------------------------------------------------------------------

@dataclass
class DisruptionEvent:
    """
    A supply chain disruption event used to seed the simulation.

    Parameters
    ----------
    name           : Human-readable event name, e.g. "Taiwan Earthquake 2024"
    ground_zero    : List of node names directly impacted (score = initial_shock)
    initial_shock  : Float [0–1]. Severity at ground-zero nodes.
                     1.0 = complete shutdown, 0.5 = 50% capacity loss.
    category       : DisruptionCategory enum value
    description    : Free-text description for the risk report
    affected_region: Optional region node name to auto-expand ground_zero
                     (all nodes located_in this region also get initial shock)
    metadata       : Extra key-value pairs carried through to the risk report
    """
    name:            str
    ground_zero:     list[str]
    initial_shock:   float               = 1.0
    category:        DisruptionCategory  = DisruptionCategory.NATURAL_DISASTER
    description:     str                 = ""
    affected_region: Optional[str]       = None
    metadata:        dict                = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.initial_shock <= 1.0:
            raise ValueError(
                f"initial_shock must be in [0, 1], got {self.initial_shock}"
            )
        if not self.ground_zero:
            raise ValueError("ground_zero must contain at least one node name.")

    @property
    def severity(self) -> SeverityLevel:
        return SeverityLevel.from_score(self.initial_shock)

    def __repr__(self) -> str:
        return (
            f"DisruptionEvent({self.name!r}, "
            f"shock={self.initial_shock:.2f}, "
            f"category={self.category.value}, "
            f"ground_zero={self.ground_zero})"
        )


# ---------------------------------------------------------------------------
# Pre-built scenario library
# Reference these in notebooks or evaluation scripts for reproducibility.
# ---------------------------------------------------------------------------

SCENARIO_LIBRARY: dict[str, DisruptionEvent] = {

    "taiwan_earthquake": DisruptionEvent(
        name="Taiwan Earthquake 2024",
        ground_zero=["TSMC", "Taiwan"],
        initial_shock=0.85,
        category=DisruptionCategory.NATURAL_DISASTER,
        description=(
            "A 7.4 magnitude earthquake strikes Taiwan, damaging TSMC fabs "
            "and disrupting semiconductor production for an estimated 3 months."
        ),
        affected_region="Taiwan",
        metadata={"duration_weeks": 12, "gdp_impact_usd_bn": 15},
    ),

    "shanghai_port_closure": DisruptionEvent(
        name="Shanghai Port Closure",
        ground_zero=["Port of Shanghai"],
        initial_shock=0.90,
        category=DisruptionCategory.LOGISTICS,
        description=(
            "The Port of Shanghai closes for 6 weeks due to a COVID-19 "
            "lockdown, halting exports from eastern China."
        ),
        metadata={"duration_weeks": 6, "teu_capacity_lost_pct": 90},
    ),

    "congo_cobalt_strike": DisruptionEvent(
        name="Congo Cobalt Strike",
        ground_zero=["Glencore"],
        initial_shock=0.60,
        category=DisruptionCategory.INDUSTRIAL,
        description=(
            "A prolonged labour strike at Glencore's Katanga Mining Complex "
            "reduces cobalt output by 60%, squeezing battery supply chains."
        ),
        metadata={"duration_weeks": 8, "cobalt_output_reduction_pct": 60},
    ),

    "red_sea_disruption": DisruptionEvent(
        name="Red Sea Shipping Disruption",
        ground_zero=["Maersk", "Port of Rotterdam"],
        initial_shock=0.55,
        category=DisruptionCategory.GEOPOLITICAL,
        description=(
            "Houthi attacks on commercial shipping force carriers to reroute "
            "around the Cape of Good Hope, adding 2 weeks and 20% cost to "
            "Asia-Europe trade lanes."
        ),
        metadata={"duration_weeks": 26, "shipping_cost_increase_pct": 20},
    ),

    "asml_export_ban": DisruptionEvent(
        name="ASML Export Restriction",
        ground_zero=["ASML"],
        initial_shock=0.70,
        category=DisruptionCategory.GEOPOLITICAL,
        description=(
            "New export control regulations restrict ASML from shipping EUV "
            "lithography machines to certain markets, constraining advanced "
            "chip production capacity."
        ),
        metadata={"duration_weeks": 52, "affected_markets": ["China"]},
    ),
}

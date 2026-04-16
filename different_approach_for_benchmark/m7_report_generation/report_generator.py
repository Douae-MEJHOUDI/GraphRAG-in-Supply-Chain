import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ImpactReport:
    query:          str
    intent_type:    str
    query_entity:   str
    context_text:   str
    llm_response:   str
    confidence:     str
    evidence_count: int
    error:          str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# Supply Chain Impact Report",
            f"**Query:** {self.query}",
            f"**Confidence:** {self.confidence}  |  "
            f"**Evidence sources:** {self.evidence_count}",
            "",
            self.llm_response,
        ]
        if self.error:
            lines.append(f"\n> ⚠ Warning: {self.error}")
        return "\n".join(lines)


SYSTEM_PROMPT = """You are a senior supply chain risk analyst specializing in the global electronics and semiconductor industry.

You will be given a structured knowledge graph context extracted from:
- SEC 10-K / 20-F filings (supplier relationships, geographic concentrations)
- USGS Mineral Commodity Summaries 2024 (country-level mineral production)
- iFixit teardown guides (product-level component dependencies)
- GDELT geopolitical event database (conflict, sanctions, export controls)

Your task is to produce a structured impact report answering the user's "what-if" question.

Rules:
1. Base every claim on the graph context. If the context lacks data for a point, say "Data not available in graph."
2. Name specific companies, countries, minerals, and products — never say "some companies" if you have names.
3. Use the Goldstein scale values to calibrate severity: below -5 = severe, -3 to -5 = significant, above -3 = moderate.
4. Express cascade effects as dependency chains: A → B → C → end impact.
5. Be concise but complete. Each section should have 2-5 bullet points.

Output format — use exactly these section headers:

## Executive Summary
[2-3 sentence overview of the disruption and its scale]

## Directly Affected Entities
[Bullet list: companies, products, countries immediately impacted]

## Cascade Effects
[Dependency chain bullet points: show the propagation path]

## Mineral & Resource Dependencies
[Only if minerals are in the context — production volumes, concentration risk]

## Geopolitical Risk Signals
[Risk events from the graph — dates, event labels, Goldstein scores]

## Alternative Suppliers & Mitigations
[Who else can fill the gap? What does the graph show about substitutes?]

## Evidence & Sources
[Bullet list of the actual evidence snippets from the graph context]

## Confidence Assessment
[High / Medium / Low — justify based on number and quality of sources]"""


def _build_user_message(query: str, context: str) -> str:
    return f"""Knowledge graph context:

{context}

---

User question: {query}

Generate the impact report following the format in your instructions."""


def _score_confidence(subgraph: dict) -> tuple[str, int]:
    supply  = len(subgraph.get("supply_edges", []))
    deps    = len(subgraph.get("depends_edges", []))
    prod    = len(subgraph.get("produces_edges", []))
    events  = len(subgraph.get("risk_events", []))
    total   = supply + deps + prod + events

    if total >= 10:
        return "High", total
    elif total >= 4:
        return "Medium", total
    else:
        return "Low", total


class ReportGenerator:

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(self, query: str, subgraph: dict, context_text: str) -> ImpactReport:
        confidence, evidence_count = _score_confidence(subgraph)

        if evidence_count == 0:
            return ImpactReport(
                query=query,
                intent_type=subgraph.get("intent_type", ""),
                query_entity=subgraph.get("query_entity", ""),
                context_text=context_text,
                llm_response="No graph data was found for this query. The entity may not exist in the knowledge graph, or there are no recorded relationships for it.",
                confidence="Low",
                evidence_count=0,
                error="No graph data retrieved.",
            )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": _build_user_message(query, context_text),
                }],
            )
            llm_text = response.content[0].text
        except Exception as ex:
            logger.error(f"[M7] LLM call failed: {ex}")
            return ImpactReport(
                query=query,
                intent_type=subgraph.get("intent_type", ""),
                query_entity=subgraph.get("query_entity", ""),
                context_text=context_text,
                llm_response=f"LLM generation failed: {ex}",
                confidence="Low",
                evidence_count=evidence_count,
                error=str(ex),
            )

        return ImpactReport(
            query=query,
            intent_type=subgraph.get("intent_type", ""),
            query_entity=subgraph.get("query_entity", ""),
            context_text=context_text,
            llm_response=llm_text,
            confidence=confidence,
            evidence_count=evidence_count,
        )

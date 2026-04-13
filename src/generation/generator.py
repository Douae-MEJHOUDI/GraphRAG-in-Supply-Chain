"""
generator.py
------------
RiskReportGenerator — the LLM generation layer for Module 4.

Primary backend — Anthropic Claude (slide 16: "Claude Sonnet 4.6")
  Requires: pip install anthropic
  Requires: ANTHROPIC_API_KEY environment variable
  Model: claude-sonnet-4-6

Secondary backends (kept for local / offline use):
  Backend B — OpenAI API
    Requires: pip install openai
    Requires: OPENAI_API_KEY environment variable

  Backend C — Ollama (local, free)
    Requires: ollama installed + model pulled
    https://ollama.ai/download

All backends receive the same prompt built by SimulationEngine and return
a RiskReport dataclass with structured fields parsed from the raw LLM output.

Structured parsing
------------------
The prompt instructs the LLM to wrap its output in XML-like section tags:

  <critical_entities>...</critical_entities>
  <dependency_chains>...</dependency_chains>
  <critical_edges>...</critical_edges>
  <mitigations>...</mitigations>
  <resilience_assessment>...</resilience_assessment>

The parser extracts each section into a dedicated field on RiskReport.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from src.simulation.engine import SimulationResult
from src.simulation.events import DisruptionEvent


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class LLMBackend(str, Enum):
    ANTHROPIC = "anthropic"   # primary — Claude Sonnet 4.6
    OPENAI    = "openai"      # secondary
    OLLAMA    = "ollama"      # local fallback


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    """
    Structured risk report produced by the LLM from the enriched context.

    Attributes
    ----------
    event_name           : name of the disruption event
    query                : the question that was asked
    full_text            : complete raw LLM output
    critical_entities    : parsed list of most exposed entities with scores
    dependency_chains    : parsed dependency path explanations
    critical_edges       : parsed single-source dependency warnings
    mitigations          : parsed mitigation recommendations
    resilience_assessment: parsed overall supply chain resilience verdict
    model                : which LLM model was used
    backend              : which backend was used
    generation_time_s    : wall-clock seconds for the LLM call
    tokens_used          : token count if available from API response
    """
    event_name:            str
    query:                 str
    full_text:             str
    critical_entities:     str = ""
    dependency_chains:     str = ""
    critical_edges:        str = ""
    mitigations:           str = ""
    resilience_assessment: str = ""
    model:                 str = ""
    backend:               str = ""
    generation_time_s:     float = 0.0
    tokens_used:           int   = 0

    def save(self, path: str | Path) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_name":            self.event_name,
            "query":                 self.query,
            "model":                 self.model,
            "backend":               self.backend,
            "generation_time_s":     round(self.generation_time_s, 2),
            "tokens_used":           self.tokens_used,
            "critical_entities":     self.critical_entities,
            "dependency_chains":     self.dependency_chains,
            "critical_edges":        self.critical_edges,
            "mitigations":           self.mitigations,
            "resilience_assessment": self.resilience_assessment,
            "full_text":             self.full_text,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Generator] Report saved → {path}")

    def display(self) -> str:
        sep = "=" * 65
        lines = [
            sep,
            f"RISK REPORT — {self.event_name}",
            f"Model: {self.model} ({self.backend})  |  "
            f"Time: {self.generation_time_s:.1f}s",
            sep,
        ]
        if self.critical_entities:
            lines += ["", "CRITICAL ENTITIES", "-" * 40, self.critical_entities]
        if self.dependency_chains:
            lines += ["", "DEPENDENCY CHAINS", "-" * 40, self.dependency_chains]
        if self.critical_edges:
            lines += ["", "CRITICAL EDGES", "-" * 40, self.critical_edges]
        if self.mitigations:
            lines += ["", "MITIGATIONS", "-" * 40, self.mitigations]
        if self.resilience_assessment:
            lines += ["", "RESILIENCE ASSESSMENT", "-" * 40,
                      self.resilience_assessment]
        if not any([self.critical_entities, self.dependency_chains,
                    self.mitigations, self.resilience_assessment]):
            lines += ["", self.full_text]
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder suffix
# ---------------------------------------------------------------------------

STRUCTURED_PROMPT_SUFFIX = """

Please structure your response using these exact XML tags so it can be parsed:

<critical_entities>
List the top 5 most critically exposed entities with their disruption scores.
One entity per line. Include the dependency path that creates the exposure.
</critical_entities>

<dependency_chains>
For each critical entity, describe the full dependency chain from the
disruption source to that entity. Use arrow notation: A → B → C.
</dependency_chains>

<critical_edges>
List all CRITICAL single-source dependency edges (marked *** CRITICAL in the
context). Explain why each is dangerous and what makes it irreplaceable.
</critical_edges>

<mitigations>
Provide 3 concrete, actionable mitigation recommendations grounded in the
graph structure. Reference specific alternative suppliers or routes from
the context where available.
</mitigations>

<resilience_assessment>
One paragraph assessing the overall supply chain resilience for this event.
Include a qualitative risk rating: LOW / MODERATE / HIGH / CRITICAL.
</resilience_assessment>
"""


# ---------------------------------------------------------------------------
# RiskReportGenerator
# ---------------------------------------------------------------------------

class RiskReportGenerator:
    """
    Generates structured risk reports using Claude Sonnet 4.6 (primary)
    or OpenAI / Ollama (secondary).

    Usage — Anthropic (default, matches slide 16):
    >>> gen = RiskReportGenerator()          # backend="anthropic" by default
    >>> report = gen.generate(sim_result)
    >>> print(report.display())

    Usage — OpenAI:
    >>> gen = RiskReportGenerator(backend="openai", model="gpt-4o-mini")

    Usage — Ollama (local, no API key):
    >>> gen = RiskReportGenerator(backend="ollama", model="mistral")

    Parameters
    ----------
    backend     : "anthropic" (default) | "openai" | "ollama"
    model       : model name string
    max_tokens  : maximum tokens for the LLM response
    temperature : sampling temperature (0.0 = deterministic)
    """

    def __init__(
        self,
        backend:     str   = "anthropic",
        model:       str   = "claude-sonnet-4-6",
        max_tokens:  int   = 1500,
        temperature: float = 0.2,
    ):
        self.backend     = LLMBackend(backend)
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

        if self.backend == LLMBackend.ANTHROPIC:
            self._validate_anthropic()
        elif self.backend == LLMBackend.OPENAI:
            self._validate_openai()
        elif self.backend == LLMBackend.OLLAMA:
            self._validate_ollama()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, sim_result: SimulationResult) -> RiskReport:
        """
        Generate a structured risk report from a SimulationResult.
        Uses Claude Sonnet 4.6 by default (slide 16).
        """
        full_prompt = sim_result.risk_report_prompt + STRUCTURED_PROMPT_SUFFIX

        print(f"[Generator] Calling {self.model} ({self.backend.value})...")
        t0 = time.time()

        raw_text, tokens = self._dispatch(full_prompt)

        elapsed = time.time() - t0
        print(f"[Generator] Done in {elapsed:.1f}s  ({tokens} tokens)")

        return self._parse_output(
            raw_text=raw_text,
            event_name=sim_result.event.name,
            query=sim_result.graphrag_result.query,
            tokens=tokens,
            elapsed=elapsed,
        )

    def _raw_generate(self, prompt: str) -> str:
        """
        Send a custom prompt and return raw text.
        Used by PipelineEvaluator for ablation conditions A and B.
        """
        raw_text, _ = self._dispatch(prompt)
        return raw_text

    def generate_batch(
        self,
        sim_results: list[SimulationResult],
        delay_s: float = 1.0,
    ) -> list[RiskReport]:
        reports = []
        for i, sim in enumerate(sim_results):
            print(f"\n[Generator] Scenario {i+1}/{len(sim_results)}: {sim.event.name}")
            reports.append(self.generate(sim))
            if i < len(sim_results) - 1:
                time.sleep(delay_s)
        return reports

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, prompt: str) -> tuple[str, int]:
        if self.backend == LLMBackend.ANTHROPIC:
            return self._call_anthropic(prompt)
        if self.backend == LLMBackend.OPENAI:
            return self._call_openai(prompt)
        return self._call_ollama(prompt)

    # ------------------------------------------------------------------
    # Anthropic backend (primary — Claude Sonnet 4.6)
    # ------------------------------------------------------------------

    def _validate_anthropic(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
            )

    def _call_anthropic(self, prompt: str) -> tuple[str, int]:
        """Call the Anthropic Messages API and return (text, total_tokens)."""
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=(
                "You are a senior supply chain risk analyst. "
                "You reason carefully from graph-structured evidence "
                "and always cite specific entity names and dependency "
                "paths in your analysis."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = message.content[0].text if message.content else ""
        tokens   = (message.usage.input_tokens + message.usage.output_tokens
                    if message.usage else 0)
        return raw_text, tokens

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    def _validate_openai(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    def _call_openai(self, prompt: str) -> tuple[str, int]:
        import openai
        client   = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a senior supply chain risk analyst. "
                    "You reason carefully from graph-structured evidence."
                )},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        raw_text = response.choices[0].message.content or ""
        tokens   = response.usage.total_tokens if response.usage else 0
        return raw_text, tokens

    # ------------------------------------------------------------------
    # Ollama backend
    # ------------------------------------------------------------------

    def _ollama_base_url(self) -> str:
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def _validate_ollama(self) -> None:
        import urllib.request, urllib.error, json as _json
        base_url = self._ollama_base_url()
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=4) as resp:
                data = _json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Ollama server not reachable at {base_url}. "
                f"Run: ollama pull {self.model}"
            ) from exc

        available = {m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)}
        accepted  = {self.model, f"{self.model}:latest"}
        if available and accepted.isdisjoint(available):
            raise RuntimeError(
                f"Ollama model '{self.model}' not available. "
                f"Run: ollama pull {self.model}"
            )

    def _call_ollama(self, prompt: str) -> tuple[str, int]:
        import urllib.request, urllib.error, json as _json
        base_url = self._ollama_base_url()
        payload  = _json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed HTTP {exc.code}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Could not connect to Ollama at {base_url}.") from exc

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", ""), data.get("eval_count", 0)

    # ------------------------------------------------------------------
    # Output parser
    # ------------------------------------------------------------------

    def _parse_output(
        self,
        raw_text:   str,
        event_name: str,
        query:      str,
        tokens:     int,
        elapsed:    float,
    ) -> RiskReport:
        def extract_tag(text: str, tag: str) -> str:
            match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        return RiskReport(
            event_name=event_name,
            query=query,
            full_text=raw_text,
            critical_entities=extract_tag(raw_text, "critical_entities"),
            dependency_chains=extract_tag(raw_text, "dependency_chains"),
            critical_edges=extract_tag(raw_text, "critical_edges"),
            mitigations=extract_tag(raw_text, "mitigations"),
            resilience_assessment=extract_tag(raw_text, "resilience_assessment"),
            model=self.model,
            backend=self.backend.value,
            generation_time_s=elapsed,
            tokens_used=tokens,
        )

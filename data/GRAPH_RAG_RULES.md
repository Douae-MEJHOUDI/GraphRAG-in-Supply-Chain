# GraphRAG Modeling Rules (Working Spec)

## Goal

Build a supply-chain graph where:
- the extractor writes only local facts (2-node edges),
- the graph engine handles multi-hop propagation,
- location-based events can spread to all impacted entities.

## What We Are Implementing

1. Atomic extraction only.
- LLM extracts one-hop triples only.
- No transitive inference in extraction (`A->C` is not created unless explicitly stated).

2. Multi-hop impact from graph semantics.
- Chains like `War in Taiwan -> TSMC -> NVIDIA` are produced by propagation over edges.
- The LLM does not pre-compute chain impacts.

3. Location-aware propagation.
- `located_in` edges are mandatory when explicit.
- If an event is linked to a location, propagation can seed all entities in that location tree.

4. Directional edge rules.
- `supplies`: supplier -> customer/dependent entity.
- `depends_on`: dependent entity -> dependency.
- `affected_by`: impacted entity -> disruption event.
- `located_in`: entity -> location.

5. Evidence and confidence on every extracted edge.
- Each edge stores short evidence text.
- Each edge stores confidence in `[0,1]`.

6. Benchmark-first workflow.
- Use edge-case passages and gold triples before full corpus extraction.
- Track precision/recall/F1 + critical checks (direction, location, disruption links).

## Extraction Contract (Current)

7. Allowed node types.
- `Supplier`, `Manufacturer`, `Part`, `Port`, `Region`, `LogisticsRoute`, `DisruptionEvent`, `Customer`.

8. Allowed relations.
- `supplies`, `depends_on`, `located_in`, `affected_by`, `ships_through`, `connects`, `alternative_to`, `sells_to`.

9. Canonical naming policy.
- Normalize company names to one canonical company identity when possible.
- Keep site/facility nodes separate if operationally distinct (example: `TSMC` vs `TSMC Arizona`).
- Store aliases in metadata for retrieval and disambiguation.

## New Rule (Requested): OR Supplier Semantics

10. Alternatives should model substitutable options for the same dependency slot, not only pairwise similarity.

Your intent (rephrased):
- If buyer `Y` can choose supplier `A` OR supplier `B`, both should connect to `Y`.
- The graph must know they are competitors/substitutes for the same requirement, so rerouting is possible during disruption.

Recommended encoding:
- Keep direct dependency edges:
  - `Y --depends_on--> A`
  - `Y --depends_on--> B`
- Add shared option-group metadata on those edges:
  - `option_group_id` (same value for both edges),
  - `primary` (true/false),
  - `switch_penalty` / `lead_time_days`,
  - `competitors` (optional list, your requested idea),
  - `evidence`, `confidence`.

Why this works:
- Propagation stays graph-native and deterministic.
- LLM can still read evidence text, but substitution logic is not only hidden in text.
- We can algorithmically reroute from failed primary supplier to backup suppliers in same `option_group_id`.

Optional stronger version (later):
- Reify dependency as a node (`DependencySlot`) to represent edge-to-edge logic cleanly.

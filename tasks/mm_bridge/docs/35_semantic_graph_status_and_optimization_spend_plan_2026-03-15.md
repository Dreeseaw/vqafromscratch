# Semantic Graph Status And Optimization Spend Plan - 2026-03-15

## Purpose

This note records what the research tracker semantic graph currently does, what it still gets wrong, and the current plan for adding optimization-loop spend attribution.

The graph is useful now as a debugging and semantic-organization tool. It is not yet the full "idea economics" view that the project wants.

## Current Semantic Graph: What Exists

The tracker now has an `Idea Graph` panel in `tracker/research` with an on-demand backend pipeline exposed through `POST /api/ideas/tree`.

Current behavior:

- builds a task-wide evidence pack from the full tracker snapshot, not just a few recent docs
- splits docs into heading-level snippets
- builds run-family summaries from tracker runs
- runs a 3-stage Codex pipeline:
  - harvest candidate concepts
  - attach/prune evidence
  - synthesize the final graph
- renders a semantic DAG-like view with anchors, nodes, and labeled edges
- shows hover popovers for node detail and edge reasoning
- exposes intermediate debug panels for:
  - evidence docs
  - snippets
  - run families
  - harvested candidates
  - kept candidates
  - dropped candidates
  - per-stage timings

This is already materially better than the first graph version, which behaved more like a progress or decision tree.

## Current Shape And Limits

The present graph is still missing three important pieces:

1. A mandatory top root
- There is no guaranteed `Starting Thoughts` node yet.
- The top of the graph is currently anchor-selected from validated concepts rather than synthesized from the original task framing.

2. Timeline flavor
- The graph is currently semantic-first but mostly time-agnostic.
- This makes it harder to read how ideas emerged, shifted, or got displaced over time.

3. Optimization-loop resource attribution
- The graph currently reasons over docs and run families.
- It does not yet attribute Codex token spend, prompt effort, or thread-level iteration cost onto ideas.

## Hardened Direction For The Graph

### 1. Mandatory Root Node

The graph should always start with a synthetic root node such as `Starting Thoughts`.

That node should be built from:

- the task description
- earliest context docs
- earliest architecture and planning docs

This root is not a normal harvested candidate. It is a pinned structural node that gives the graph a stable top-level origin.

### 2. Timeline-Flavored Layout

Time should influence layout, but not dominate graph truth.

Planned node metadata:

- `firstSeenAt`
- `lastSeenAt`
- `phase`
- `timeConfidence`

Planned layout policy:

- root pinned at the top
- horizontal ordering roughly tracks first appearance or phase
- vertical structure stays semantic
- cross-links and backward links remain allowed

The point is human readability. The graph should communicate "how ideas developed" without pretending all reasoning is strictly chronological.

## Optimization Spend Attribution: Actual Goal

The important resource is not only run-side GPU time.

The deeper goal is to measure optimization-loop spend from the auto-research process itself:

- Codex tokens spent over time
- prompts and threads used to resolve a research bottleneck
- tool activity and file touches associated with that work
- later downstream ideas that benefited from that earlier optimization effort

Example:

- a large amount of token spend might go into making KV-cache eval both correct and fast
- later Hammer-family runs benefit from that work
- those later runs are therefore trading training resources for earlier optimization-loop spend

That spend is economically meaningful even if local usage happens under a flat subscription. Tokens can still be tracked and later converted into a notional API-equivalent cash estimate if needed.

## Planned Spend Model

The spend model should be separate from run-resource accounting.

Two spend types should be attached to graph ideas:

1. `direct optimization spend`
- tokens and thread effort spent explicitly working on that idea

2. `inherited optimization spend`
- upstream enabling work that made the idea or run family viable

These should not be collapsed into one number. Direct and inherited spend tell different stories.

## Recommended Architecture

### A. Preprocess Codex Logs Into A Stable Local Ledger

This should be a separate Bun CLI job, similar in spirit to the existing log stitcher.

Raw sources:

- `~/.codex/sessions/**/*.jsonl`
- `~/.codex/history.jsonl`

The ingest step should emit normalized local records such as:

- `sessionId`
- `startAt`
- `endAt`
- token deltas
- prompt excerpt or hash
- tool calls
- files touched
- docs mentioned
- run ids or run families mentioned
- task or cwd hints

This step should be mostly append-only and incremental. It should not depend on the current semantic graph.

### B. Join Spend Onto Kept Ideas After Evidence Attachment

Semantic join should happen after candidate pruning, not before.

Reason:

- by then the graph pipeline has canonical idea labels
- aliases are already collapsed
- evidence refs already exist
- dropped noise has already been filtered out

This makes attribution more trustworthy than joining against raw prompts or unstable early candidates.

Join signals should include:

- label and alias overlap
- shared docs and snippets
- shared run ids or run families
- touched files
- timing proximity

### C. Propagate Inherited Spend Over The Final Graph

Once direct attribution exists, inherited spend can be rolled forward over graph dependencies.

Important interpretation rule:

- direct spend is additive accounting
- inherited spend is contextual dependency cost

Inherited spend will often appear in multiple downstream nodes. That is correct for context, but it should not be mistaken for a global deduplicated total.

## Why Preprocessing Matters

The current semantic graph pipeline already spends a noticeable amount of Codex budget.

Because of that, optimization-spend attribution should not add more expensive prompting into the hot path by default.

Preferred policy:

- preprocess raw Codex history once
- cache it locally
- use deterministic or near-deterministic joins during graph generation
- reserve any extra model-assisted adjudication for optional offline cleanup, not the default path

This keeps the graph useful without creating a second expensive graph-of-the-graph process.

## Debuggability Requirements

Spend attribution will be easy to mistrust unless the provenance stays visible.

For every node, the tracker should eventually be able to show:

- which session spans were attached
- why they were attached
- which prompts or files were involved
- which spend was direct versus inherited
- what spend remained unattributed

The existing graph debug panels are a good foundation for this. The spend system should follow the same style: explicit artifacts, not hidden scoring magic.

## Short-Term Implementation Order

1. Add the mandatory root node and soft time metadata to the graph schema and layout.
2. Build a Codex-spend ingest CLI that materializes a normalized local ledger.
3. Attach that ledger to non-dropped ideas after evidence attachment.
4. Roll up direct and inherited optimization spend into graph nodes and debug panels.

## Practical Summary

The semantic graph is now real, multi-stage, and evidence-backed, but it is still missing origin, temporal flavor, and idea-level optimization economics.

The next major step should not be "prompt harder." It should be:

- stable root
- soft time
- preprocessed Codex spend ledger
- semantic join onto validated ideas

That is the path from "semantic graph" to "research process graph."

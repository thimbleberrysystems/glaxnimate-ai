# glaxnimate-ai

An MCP interface that lets an LLM author 2D cartoon animation with Glaxnimate.

## What this project actually is

Not a thin wrapper over Glaxnimate's API. Two constraints drive the design:

1. **An LLM cannot draw.** So the repo ships a *procedural rigged cartoon library*
   the model **parameterises** rather than draws — a generic joint-graph rig plus a
   phase-offset gait engine. A human is one preset among many (quadruped, bird,
   vehicle, blob). **Scope is "animate anything", not "animate people"** — if a
   change only works for humanoids, it is wrong.
2. **An LLM animating blind produces garbage — but most of the fix is arithmetic,
   not vision.** Feedback is a tiered critic stack, cheapest first: a free linter
   (contact slip, joint integrity, bounds) → numeric diagnostics (spacing charts,
   arc curvature, silhouette, balance) → rendered images only for what numbers
   cannot judge → the human. Images cost ~1,400 tokens; reach for them last.

The MCP server is the smallest part. The library and the feedback loop are the product.

## Engine setup

**Do not `pip install glaxnimate`.** The PyPI wheel is tagged `py3-none` but is
built for CPython 3.13 on **openSUSE, against SUSE's patched ffmpeg** — it demands
symbols (`LIBAVCODEC_61.19_SUSE`) that no other ffmpeg exports. It installs
anywhere and imports nowhere.

Build the bindings from source instead:

```sh
bash scripts/setup.sh          # needs sudo (apt); builds the GUI app + bindings
```

This installs the GUI to `/usr/local/bin/glaxnimate` and builds
`glaxnimate.cpython-314-*.so`, which is symlinked into `.venv`. Regenerate the
API reference after any rebuild:

```sh
.venv/bin/python scripts/introspect_api.py   # -> docs/glaxnimate-api.md
```

## API reference

**`docs/glaxnimate-api.md` is the source of truth — the online docs are a version
behind and will mislead you** (they still show `document.main`, which no longer
exists). The worst trap: `Layer.animation.last_frame` defaults to `-1`, so a layer
is invisible and every frame renders blank, with no error.

## Commits

Plain, conventional messages describing the change, authored by Thimble Berry.
**No AI attribution of any kind** — no `Co-Authored-By: Claude`, no
"Generated with Claude Code" footer in PRs.

## Using it from Claude

```sh
claude mcp add glaxnimate -- /home/franklynece/glaxnimate-ai/.venv/bin/python \
    -m glaxnimate_ai.mcp.server
```

Then just ask: *"animate a man walking home from school"*.

The tools are ordered so the cheap tiers come first — `lint_animation` (free) and
`diagnose_animation` (~500 tokens, names the frame) before `render_contact_sheet`
(~1,400 tokens, says "hmm"). Pushing the model down that ladder is the point.

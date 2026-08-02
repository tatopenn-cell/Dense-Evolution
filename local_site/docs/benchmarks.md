# Benchmarks

Included directly from the repository [`README.md`](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/README.md)
so it never goes stale relative to the single source of truth. See the
[Changelog](changelog.md) for how these numbers evolved release to release.

## Anti-OOM chunking vs. PennyLane, and general benchmarks

> The distributed-dispatch section below covers `run_chunk_distributed` -- multi-device
> sharding, not a benchmark number itself, but kept here since it's part of the same
> performance story. General throughput/drift numbers (measured on Google Colab Free Tier)
> follow further down.

{% include-markdown "../README.md" start="### Benchmark vs PennyLane — Windows CPU (8 GB RAM)" end="## ▍ Dashboard Panels" %}

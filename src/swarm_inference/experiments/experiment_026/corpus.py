"""Frozen development and held-out text, materialized before any model runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .io import digest, write_once


def build_corpus(root: Path) -> None:
    tasks = [
        ("factual", "short", "Write a detailed textbook explanation of the water cycle, including evaporation, condensation, precipitation, groundwater, glaciers, and the role of the Sun. Use complete paragraphs and at least 700 words."),
        ("code", "short", "Implement a self-contained Python module for a bounded LRU cache with get, put, deletion, iteration, expiry using an injected monotonic clock, and examples. Explain invariants and edge cases after the code. Provide a complete implementation, not a sketch."),
        ("reasoning", "short", "A warehouse has bins of capacities 7, 11, and 13 litres, initially empty. An unlimited tap and drain are available. Work through a rigorous way to measure exactly 5 litres. Explain each transfer, the state after each step, and why the method works. Then give a general breadth-first search algorithm for solving arbitrary capacities."),
        ("structured", "2k", "Continue this structured inventory with records 101 through 180. Each record has the form 'Record N: owner=team-(N modulo 7); replicas=3; status=ready; checksum=pending'. Output every record, one per line, without omissions or ellipses."),
        ("generation", "2k", "Write a 1000-word short story about a botanist cataloguing seeds aboard a generation ship. Incorporate the reference notes naturally, and give the story a concrete ending. Start directly with the story."),
        ("retrieval", "8k", "Using only the register below, list the access codes for entries 20 through 99 in increasing numerical order, one per line. Then explain how you checked the order. Do not infer codes; copy each exact value."),
    ]
    heldout = [
        ("factual", "short", "Write a detailed 1200-word explanation of how a municipal drinking-water treatment system works, from intake to distribution. Include sedimentation, filtration, disinfection, measurement, and maintenance. State uncertainty where appropriate."),
        ("code", "2k", "Write a complete Python implementation of a dependency-aware job scheduler for a directed acyclic graph, including cycle detection, deterministic topological ordering, bounded concurrency, cancellation, and a worked example. Discuss correctness invariants."),
        ("retrieval", "8k", "Copy register entries 130 through 209 with their access codes, one per line in order. Then write a paragraph explaining your verification procedure. Use only the register below."),
    ]
    for split, items in (("development", tasks), ("sealed", heldout)):
        rows = []
        for index, (category, context, task) in enumerate(items):
            prefix = ""
            if context == "2k":
                prefix = "Reference notes:\n" + "\n".join(
                    f"Note {n}: Team {(n + index) % 7} catalogued sample {n * 13 + 17}; the archive retains a dated record and a separate verification copy."
                    for n in range(1, 75)) + "\n\n"
            elif context == "8k":
                prefix = "Register:\n" + "\n".join(
                    f"Entry {n:03d}: access_code={hashlib.sha256(f'e026:{split}:{n}'.encode()).hexdigest()[:12]}; owner=unit-{n % 11}; checked=yes."
                    for n in range(1, 261)) + "\n\n"
            content = prefix + task
            rows.append({"prompt_id": f"{split}-{index + 1:02d}-{category}",
                         "split": split, "category": category, "context_class": context,
                         "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                         "generation_limit": 256 if split == "development" else 640})
        write_once(root / "corpus" / f"{split}.json", {"schema_version": 1, "prompts": rows, "sha256": digest(rows)})


if __name__ == "__main__":
    build_corpus(Path("artifacts/experiment-026"))

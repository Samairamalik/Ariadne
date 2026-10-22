"""Entity merging: look first, then apply.

    python resolve_entities.py                 # DRY RUN: list candidate pairs with their context similarity
    python resolve_entities.py --apply         # merge pairs that pass their tier's cutoff (logged in `merges`)
    python resolve_entities.py --apply 0.7     # same, with 0.7 as the same-name cutoff
"""
import asyncio
import sys

from app import db
from app.config import settings
from app.ingestion.resolve import resolve_duplicates


async def main() -> None:
    apply = "--apply" in sys.argv
    nxt = sys.argv[sys.argv.index("--apply") + 1] if apply and len(sys.argv) > sys.argv.index("--apply") + 1 else None
    cutoff = float(nxt) if nxt else None
    await db.init_pool()
    pairs = await resolve_duplicates(cutoff, apply=apply)
    print(f"cutoffs: same-name {cutoff or settings.resolve_context_cutoff}, contained {settings.resolve_subset_cutoff}\n")
    for p in sorted(pairs, key=lambda p: (p.kind, p.similarity is None, -(p.similarity or 0))):
        sim = "  n/a" if p.similarity is None else f"{p.similarity:5.2f}"
        status = "MERGED" if p.merged else "      "
        left = f"{p.name!r} ({p.kept_type})"
        print(f"{p.kind:9s} {sim} {status}  {left}  <-  {p.other_name!r} ({p.other_type})")
    print(f"\n{len(pairs)} candidate pairs" + (f", {sum(p.merged for p in pairs)} merged" if apply else " (dry run, nothing changed)"))
    await db.close_pool()


asyncio.run(main())

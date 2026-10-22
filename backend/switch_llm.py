"""Switch the running API between offline (Ollama) and cloud (Groq -> Gemini) and prove it works.

    python switch_llm.py offline
    python switch_llm.py cloud
    python switch_llm.py            # just show the current status

The API must be running. Override its address with ARIADNE_API_URL (default http://localhost:8100).
"""
import os
import sys

import httpx

BASE = os.environ.get("ARIADNE_API_URL", "http://localhost:8100").rstrip("/")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode not in (None, "offline", "cloud"):
        print(__doc__)
        return 2
    try:
        with httpx.Client(base_url=BASE, timeout=200) as c:
            r = c.put("/api/llm", json={"mode": mode}) if mode else c.get("/api/llm")
            r.raise_for_status()
            st = r.json()
            print(f"mode: {st['mode']}   will try: {', '.join(st['will_try']) or 'nothing'}")
            if st["sends_text_off_machine"]:
                print("note: document text is sent to cloud providers in this mode")
            for name, p in st["cloud"]["providers"].items():
                print(f"  {name}: {'key set' if p['configured'] else 'no key'} ({p['model']})")
            print(f"  ollama: {st['ollama']['selected']} (installed: {', '.join(st['ollama']['installed']) or 'none'})")
            if not st["will_try"]:
                print("no provider available; add a key to .env and restart the API, or use offline")
                return 1
            t = c.post("/api/llm/test").json()
    except httpx.HTTPError as e:
        print(f"could not reach the API at {BASE}: {e}")
        return 1
    if t["ok"]:
        print(f"test ok: {t['provider']}/{t['model']} replied {t['reply']!r} in {t['latency_ms']} ms")
        return 0
    print(f"test failed: {t['error']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

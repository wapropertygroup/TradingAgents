"""Diagnose why GOOGLE_API_KEY isn't reaching the Gemini client.

Run this from the SAME shell and directory you run `tradingagents` from:

    conda activate tradingagents
    cd ~/workspace/TradingAgents
    python diag_key.py

It never prints your full key - only lengths and short prefixes.
"""

import os
import sys
from pathlib import Path


def show(label, key):
    if key is None:
        return f"{label}: <absent>"
    return (
        f"{label}: len={len(key)} head={key[:8]!r} tail={key[-4:]!r} "
        f"quotes={'YES' if chr(34) in key or chr(39) in key else 'no'} "
        f"whitespace={'YES' if key != key.strip() else 'no'}"
    )


print("=" * 68)
print("1. PROCESS CONTEXT")
print("=" * 68)
print(f"cwd:            {Path.cwd()}")
print(f"python:         {sys.executable}")
print(f"conda env:      {os.environ.get('CONDA_DEFAULT_ENV', '<none>')}")

# Read the shell-inherited values BEFORE importing tradingagents,
# because importing it runs load_dotenv().
shell_google = os.environ.get("GOOGLE_API_KEY")
shell_gemini = os.environ.get("GEMINI_API_KEY")

print()
print("=" * 68)
print("2. WHAT YOUR SHELL EXPORTED (before .env is loaded)")
print("=" * 68)
print(show("GOOGLE_API_KEY", shell_google))
print(show("GEMINI_API_KEY", shell_gemini))
if shell_google is not None:
    print()
    print("  >>> GOOGLE_API_KEY is exported by your SHELL.")
    print("  >>> load_dotenv uses override=False, so .env CANNOT replace it.")
    if not shell_google.strip():
        print("  >>> It is EMPTY. This is the bug. Fix: unset GOOGLE_API_KEY")
else:
    print()
    print("  ok - shell does not export it, so .env is free to supply it.")

print()
print("=" * 68)
print("3. WHICH .env IS FOUND")
print("=" * 68)
try:
    from dotenv import find_dotenv

    found = find_dotenv(usecwd=True)
    print(f"find_dotenv(usecwd=True) -> {found or '<NOTHING FOUND>'}")
    if not found:
        print("  >>> No .env found from this directory. Keys will be missing.")
        print("  >>> Fix: cd ~/workspace/TradingAgents")
except ImportError:
    print("python-dotenv is NOT installed -> .env is never loaded, silently.")
    print("  >>> Fix: pip install python-dotenv")

print()
print("=" * 68)
print("4. AFTER tradingagents LOADS .env")
print("=" * 68)
import tradingagents  # noqa: E402  (triggers load_dotenv)

final = os.environ.get("GOOGLE_API_KEY")
print(show("GOOGLE_API_KEY", final))
if final and shell_google and final == shell_google:
    print("  >>> value came from your SHELL (not .env)")
elif final:
    print("  >>> value came from .env")

print()
print("=" * 68)
print("5. RESOLVED CONFIG")
print("=" * 68)
from tradingagents.default_config import DEFAULT_CONFIG as cfg  # noqa: E402

for k in ("llm_provider", "deep_think_llm", "quick_think_llm"):
    print(f"{k:18} = {cfg[k]!r}")

print()
print("=" * 68)
print("6. LIVE API CALL")
print("=" * 68)
from tradingagents.llm_clients.google_client import GoogleClient  # noqa: E402

for model in (cfg["quick_think_llm"], cfg["deep_think_llm"]):
    try:
        result = GoogleClient(model).get_llm().invoke("Reply with exactly: OK")
        text = result if isinstance(result, str) else getattr(result, "content", result)
        print(f"  {model:26} -> SUCCESS {str(text).strip()[:16]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {model:26} -> FAILED  {type(exc).__name__}")
        print(f"      {str(exc)[:200]}")

print()
print("Done. Paste sections 2, 4 and 6 back if it still fails.")

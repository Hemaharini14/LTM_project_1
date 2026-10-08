"""Windows app entry point. The app's own .py files, templates, models and data
live in the 'app' folder next to this exe (plain files, so paths such as
src/../models/artifacts keep working); this exe only supplies Python + libraries.

The imports below exist only so PyInstaller bundles every library the app uses.
"""
import bleach, concurrent.futures, contextlib, dotenv, flask, functools, httpx, io, itertools  # noqa
import joblib, json, langchain_anthropic, langchain_core, langchain_openai, langgraph, markdown  # noqa
import math, numpy, pandas, pathlib, random, re, sklearn, sqlite3, threading, time, torch  # noqa
import truststore, typing, urllib.request, werkzeug, zlib, decimal, csv, hashlib, uuid, secrets, html  # noqa
import os
import sys
import threading
import webbrowser

HOST, PORT = "127.0.0.1", 5000


def main():
    exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    root = os.path.join(exe_dir, "app")
    if not os.path.isdir(root):
        root = os.path.dirname(exe_dir)  # running from source
    os.chdir(root)
    sys.path[:0] = [root, os.path.join(root, "src")]
    os.environ.setdefault("SECRET_KEY", "smartroute-local-desktop-key")
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(root, ".env"))
    except Exception:
        pass
    import api
    print(f"SmartRouteAI running at http://{HOST}:{PORT}  (close this window to quit)")
    threading.Timer(3, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    api.app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()

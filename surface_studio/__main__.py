from __future__ import annotations

import argparse
import threading
import webbrowser

from .server import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Surface attachment design workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    httpd = create_server(args.host, args.port)
    print(f"Surface Studio running at {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        def open_browser():
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Timer(0.6, open_browser).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Surface Studio.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

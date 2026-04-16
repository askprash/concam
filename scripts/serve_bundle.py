#!/usr/bin/env python3
"""Simple HTTP server with Range-request support for serving labeler bundles.

Python 3.8's built-in http.server does not handle the Range header, so
browsers cannot seek large video files.  This script adds proper
206 Partial Content responses so the HTML5 video element can seek freely.

Usage:
    python3 scripts/serve_bundle.py <bundle_dir> [port]
    # default port: 8080
"""

import http.server
import mimetypes
import os
import sys
from pathlib import Path


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler extended with HTTP Range request support."""

    def send_head(self):
        path = self.translate_path(self.path)

        # Directories: fall through to the parent (index page).
        if os.path.isdir(path):
            return super().send_head()

        # Regular files: check for a Range header.
        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")

        if range_header is None:
            # No Range header — serve the whole file (200 OK).
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f

        # Parse "Range: bytes=start-end"
        try:
            byte_range = range_header.strip().replace("bytes=", "")
            start_str, end_str = byte_range.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except (ValueError, AttributeError):
            self.send_error(416, "Requested Range Not Satisfiable")
            f.close()
            return None

        end = min(end, file_size - 1)
        if start > end or start < 0:
            self.send_error(416, "Requested Range Not Satisfiable")
            f.close()
            return None

        chunk_size = end - start + 1
        f.seek(start)

        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(chunk_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return f

    def log_message(self, fmt, *args):
        # Suppress per-request noise for video byte-range spam.
        if "206" not in args[1] if len(args) > 1 else False:
            super().log_message(fmt, *args)
        else:
            pass  # swallow 206 lines


if __name__ == "__main__":
    bundle_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    if not bundle_dir.is_dir():
        print(f"Error: {bundle_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    os.chdir(bundle_dir)
    handler = RangeHTTPRequestHandler
    with http.server.HTTPServer(("", port), handler) as httpd:
        print(f"Serving {bundle_dir.resolve()} on http://localhost:{port}/")
        print(f"Open: http://localhost:{port}/labeler.html")
        print("Ctrl-C to stop.")
        httpd.serve_forever()

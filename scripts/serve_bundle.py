#!/usr/bin/env python3
"""HTTP server with Range-request support for serving labeler bundles.

Python 3.8's built-in http.server has no Range header support, causing
browsers to buffer the entire video before playing. This script adds proper
206 Partial Content responses so the HTML5 video element can seek freely.

Usage:
    python3 scripts/serve_bundle.py <bundle_dir> [port]
    # default port: 8080
"""

import http.server
import io
import os
import sys
from pathlib import Path


_CHUNK = 256 * 1024  # 256 KB copy buffer


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def send_head(self):
        path = self.translate_path(self.path)

        if os.path.isdir(path):
            return super().send_head()

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")

        if range_header is None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            # Store sentinel so do_GET copies the whole file.
            self._range_length = None
            return f

        # Parse "Range: bytes=start-end"
        try:
            spec = range_header.strip().replace("bytes=", "")
            start_str, end_str = spec.split("-")
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

        chunk = end - start + 1
        f.seek(start)

        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(chunk))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        # Store how many bytes to send so copyfile doesn't over-run.
        self._range_length = chunk
        return f

    def copyfile(self, source, outputfile):
        """Copy exactly _range_length bytes (or all, for non-range requests)."""
        remaining = getattr(self, "_range_length", None)
        try:
            if remaining is None:
                # Full-file response — use the fast default.
                import shutil
                shutil.copyfileobj(source, outputfile, _CHUNK)
            else:
                while remaining > 0:
                    buf = source.read(min(_CHUNK, remaining))
                    if not buf:
                        break
                    outputfile.write(buf)
                    remaining -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            # Browser aborted (normal for video seeking — it opens a new
            # range request immediately). Not an error worth printing.
            pass

    def log_message(self, fmt, *args):
        # Suppress 206 noise; keep everything else.
        if len(args) >= 2 and "206" in str(args[1]):
            return
        super().log_message(fmt, *args)


if __name__ == "__main__":
    bundle_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080

    if not bundle_dir.is_dir():
        print(f"Error: {bundle_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    os.chdir(bundle_dir)
    with http.server.HTTPServer(("", port), RangeHTTPRequestHandler) as httpd:
        print(f"Serving {bundle_dir.resolve()}")
        print(f"Open:   http://localhost:{port}/labeler.html")
        print("Ctrl-C to stop.")
        httpd.serve_forever()

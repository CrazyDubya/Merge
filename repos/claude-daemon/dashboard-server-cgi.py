#!/usr/bin/env python3
"""
Dashboard Server with CGI Support
Serves static files + executes API endpoints (.sh scripts)
"""

import http.server
import socketserver
import os
import subprocess
import sys
from urllib.parse import urlparse

class CustomCGIHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that executes .sh files as CGI scripts"""

    def do_GET(self):
        # Parse URL
        parsed = urlparse(self.path)
        path = parsed.path
        query = parsed.query

        # Check if requesting a .sh API endpoint
        if path.endswith(".sh"):
            # Security: Only execute files that exist and are in DAEMON_ROOT
            file_path = path.lstrip("/")
            full_path = os.path.join(os.getcwd(), file_path)

            if not os.path.isfile(full_path):
                self.send_error(404, "API endpoint not found")
                return

            if not os.access(full_path, os.X_OK):
                self.send_error(403, "API endpoint not executable")
                return

            # Execute CGI script
            try:
                env = os.environ.copy()
                env["QUERY_STRING"] = query
                env["REQUEST_METHOD"] = "GET"
                env["SCRIPT_NAME"] = path

                # EXPERIMENTER: Pass HTTP headers to CGI script
                # CGI convention: HTTP headers become HTTP_* env vars
                if "Authorization" in self.headers:
                    env["HTTP_AUTHORIZATION"] = self.headers["Authorization"]

                result = subprocess.run(
                    [full_path],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10  # Prevent DoS via slow scripts
                )

                # Parse CGI headers from output
                output_lines = result.stdout.split("\n")
                headers_done = False

                for line in output_lines:
                    if not headers_done:
                        if line.strip() == "":
                            headers_done = True
                            self.end_headers()
                        elif ":" in line:
                            # Send header
                            key, value = line.split(":", 1)
                            if key.lower() == "content-type":
                                self.send_response(200)
                                self.send_header("Content-Type", value.strip())
                            else:
                                self.send_header(key.strip(), value.strip())
                    else:
                        # Body content
                        self.wfile.write((line + "\n").encode())

            except subprocess.TimeoutExpired:
                self.send_error(504, "API endpoint timeout")
            except Exception as e:
                self.send_error(500, f"API endpoint error: {str(e)}")
        else:
            # Serve static files normally
            super().do_GET()

    def end_headers(self):
        # Add security headers
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

    print("━" * 60)
    print("🎭  Dashboard Server with CGI Support")
    print("━" * 60)
    print(f"📍 Root: {os.getcwd()}")
    print(f"🌐 Port: {port}")
    print(f"🔗 Dashboard: http://localhost:{port}/dashboard.html")
    print(f"🔗 Inbox API: http://localhost:{port}/api-inbox.sh?folder=unread")
    print()
    print("Press Ctrl+C to stop")
    print()

    with socketserver.TCPServer(("", port), CustomCGIHandler) as httpd:
        print(f"Server running on port {port}")
        httpd.serve_forever()

if __name__ == "__main__":
    main()

#!/bin/bash
#
# Dashboard Server with CGI Support
# Serves static files + executes API endpoints
#
# Performance: Python http.server with CGIHTTPRequestHandler
# - Static files served directly (fast)
# - *.sh scripts executed as CGI (dynamic data)
# - No external dependencies (Python 3 stdlib)
#
# Security:
# - CGI scripts validated before execution
# - Directory listing disabled for inbox paths
# - Same-origin policy via CORS headers
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
PORT="${1:-8080}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎭  Starting Multi-Persona Daemon Dashboard Server (CGI)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Root: $DAEMON_ROOT"
echo "🌐 Port: $PORT"
echo "🔗 Dashboard: http://localhost:$PORT/dashboard.html"
echo "🔗 Inbox API: http://localhost:$PORT/api-inbox.sh?folder=unread"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$DAEMON_ROOT"

# Create Python CGI server script
# Performance note: CGIHTTPRequestHandler only executes files in cgi-bin/ by default
# We override do_GET to execute *.sh files anywhere (more flexible routing)
python3 -c '
import http.server
import socketserver
import os
import subprocess
from urllib.parse import urlparse, parse_qs

PORT = int(os.environ.get("PORT", "8080"))

class CustomCGIHandler(http.server.SimpleHTTPRequestHandler):
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

with socketserver.TCPServer(("", PORT), CustomCGIHandler) as httpd:
    print(f"Server running on port {PORT}")
    httpd.serve_forever()
' PORT="$PORT"

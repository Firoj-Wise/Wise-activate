import http.server
import socketserver
import mimetypes
import os

PORT = 8000
DIRECTORY = "web"

class WasmHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        # Force correct MIME type for WASM
        if self.path.endswith('.wasm'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/wasm')
            self.end_headers()
            
            # Serve file content
            file_path = os.path.join(DIRECTORY, self.path.lstrip('/'))
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
            return
        
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

# Add mappings just in case
mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('application/javascript', '.js')

print(f"Starting server on http://localhost:{PORT} (Serving '{DIRECTORY}' with WASM support)")

with socketserver.TCPServer(("", PORT), WasmHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

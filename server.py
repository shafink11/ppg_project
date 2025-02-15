from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class WebhookHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Read the length of the request body (as declared in the header)
        content_length = int(self.headers.get('Content-Length', 0))
        # Read the body itself
        post_data = self.rfile.read(content_length)
        # Convert bytes to string for printing/logging
        body = post_data.decode('utf-8')

        # Print the request body to the console
        print("=== Webhook received! ===")
        print(body)
        print("=========================")

        # Send an HTTP 200 response (OK)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Webhook received")

    # Optional: Handle GET requests to check if the server is running
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Webhook server is running")

def run_server(port=3000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, WebhookHandler)
    print(f"Serving on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()

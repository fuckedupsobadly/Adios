# Production entry point (wsgi.py) tests
# Render's free plan blocks outbound SMTP, so on the live site password-reset
# emails go through Brevo's HTTPS API instead. "Forgot password" must send the
# email there, and must still respond when the mail service fails or never answers.

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs in a separate process: importing wsgi.py turns on production-only
# settings (secure cookies, proxy headers) that must not leak into other tests.
FORGOT_PASSWORD_REQUEST = """
client = wsgi.app.test_client()
client.post("/register", data={"email": "student@nis.kz",
                               "password": "Kazakh2026",
                               "confirm_password": "Kazakh2026"})
response = client.post("/forgot-password", data={"email": "student@nis.kz"})
print(response.status_code, response.headers.get("Location"))
"""


def request_password_reset(setup_code="", **env_vars):
    """Returns the response ("302 /login"), everything the app printed and the seconds it took."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MAIL_", "BREVO_"))}
    env.update(SECRET_KEY="test-only-key", DATABASE_URL="sqlite:///:memory:", **env_vars)
    start = time.time()
    result = subprocess.run([sys.executable, "-c", "import wsgi\n" + setup_code + FORGOT_PASSWORD_REQUEST],
                            cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=60)
    seconds = time.time() - start
    assert result.returncode == 0, result.stderr
    # The app also prints "Model loaded." and any mail error, so the response is the last line
    response = result.stdout.strip().splitlines()[-1]
    print("\n  result:", response, f"in {seconds:.1f}s")
    return response, result.stdout, seconds


class FakeBrevoHandler(BaseHTTPRequestHandler):
    """Stands in for api.brevo.com: records each request and answers with server.status."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append({"path": self.path,
                                     "api_key": self.headers["api-key"],
                                     "json": json.loads(body)})
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def fake_brevo():
    server = HTTPServer(("127.0.0.1", 0), FakeBrevoHandler)
    server.status = 201
    server.requests = []
    server.setup_code = f'wsgi.BREVO_URL = "http://127.0.0.1:{server.server_port}/v3/smtp/email"\n'
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_reset_email_is_sent_through_brevo(fake_brevo):
    response, output, _ = request_password_reset(fake_brevo.setup_code, BREVO_API_KEY="test-api-key",
                                                 MAIL_USERNAME="sender@nis.kz")
    assert response == "302 /login"
    assert len(fake_brevo.requests) == 1, output
    sent = fake_brevo.requests[0]
    assert sent["path"] == "/v3/smtp/email"
    assert sent["api_key"] == "test-api-key"
    assert sent["json"]["sender"]["email"] == "sender@nis.kz"
    assert [to["email"] for to in sent["json"]["to"]] == ["student@nis.kz"]
    assert sent["json"]["subject"] == "Password Reset"
    assert "/reset-password/" in sent["json"]["textContent"]


def test_brevo_error_is_logged_not_crashed(fake_brevo):
    fake_brevo.status = 401  # e.g. a wrong API key
    response, output, _ = request_password_reset(fake_brevo.setup_code, BREVO_API_KEY="wrong-key",
                                                 MAIL_USERNAME="sender@nis.kz")
    assert response == "302 /login"
    assert len(fake_brevo.requests) == 1, output
    assert "Mail error" in output


def test_forgot_password_responds_when_mail_server_never_answers():
    # Accepts the TCP connection but never sends the SMTP greeting, like Render's blocked ports
    silent_server = socket.socket()
    silent_server.bind(("127.0.0.1", 0))
    silent_server.listen()
    try:
        response, _, seconds = request_password_reset(MAIL_SERVER="127.0.0.1",
                                                      MAIL_PORT=str(silent_server.getsockname()[1]))
    finally:
        silent_server.close()
    assert response == "302 /login"
    assert seconds < 30

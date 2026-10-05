"""
Production entry point, used by gunicorn on Render:  gunicorn wsgi:app

app.py is kept exactly as documented and tested; settings that only matter
when the site is live on the internet are added here instead.
"""
import json
import os
import socket
import urllib.request

from sqlalchemy import event, exc
from werkzeug.middleware.proxy_fix import ProxyFix

from app import app, db, mail

# The session cookie is signed with SECRET_KEY. With the public default anyone
# could forge a login, so refuse to start without a real key.
if app.config["SECRET_KEY"] == "dev-secret-change-in-prod":
    raise RuntimeError("Set the SECRET_KEY environment variable before starting in production.")

# Requests reach Flask through two proxies: Vercel (the public URL) and Render's
# load balancer. Trust their X-Forwarded-* headers so links built with
# url_for(..., _external=True), like the password reset link, use https and the
# public host.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# The site is only served over HTTPS, so never send the login cookie over plain HTTP.
app.config["SESSION_COOKIE_SECURE"] = True

# Flask-Mail opens its SMTP connection with no timeout. If the mail server never
# answers (Render's free plan silently drops outbound SMTP), "Forgot password"
# hangs the only gunicorn worker until it is killed and the whole site goes
# down. smtplib uses this default, so give up after 10 seconds instead; the
# error is then caught and logged in forgot_password().
socket.setdefaulttimeout(10)

# Render's free plan blocks SMTP entirely, so password-reset emails are sent
# through Brevo's HTTPS API instead. app.py still calls mail.send(msg); when a
# Brevo API key is set, that call goes to send_with_brevo() below.
BREVO_URL = "https://api.brevo.com/v3/smtp/email"


def send_with_brevo(msg):
    """Send a Flask-Mail Message through Brevo.

    Docs: https://developers.brevo.com/reference/send-transac-email
    forgot_password() in app.py catches any exception raised here and logs it.
    """
    email = {
        "sender": {"email": msg.sender},
        "to": [{"email": address} for address in msg.recipients],
        "subject": msg.subject,
        "textContent": msg.body,
    }
    request = urllib.request.Request(BREVO_URL, data=json.dumps(email).encode(), method="POST",
                                     headers={"api-key": os.environ["BREVO_API_KEY"],
                                              "content-type": "application/json",
                                              "accept": "application/json"})
    # urlopen raises an error for any 4xx/5xx reply (e.g. a wrong API key), so it gets logged
    urllib.request.urlopen(request, timeout=10).close()


if os.environ.get("BREVO_API_KEY"):
    mail.send = send_with_brevo


# Neon pauses idle databases and closes their connections. Test each pooled
# connection before using it, so the first request after a pause gets a fresh
# connection instead of an error.
def ping_connection(dbapi_connection, connection_record, connection_proxy):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT 1")
    except Exception:
        raise exc.DisconnectionError()
    finally:
        cursor.close()


with app.app_context():
    event.listen(db.engine, "checkout", ping_connection)

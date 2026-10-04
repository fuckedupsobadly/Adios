"""
Production entry point, used by gunicorn on Render:  gunicorn wsgi:app

app.py is kept exactly as documented and tested; settings that only matter
when the site is live on the internet are added here instead.
"""
from sqlalchemy import event, exc
from werkzeug.middleware.proxy_fix import ProxyFix

from app import app, db

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

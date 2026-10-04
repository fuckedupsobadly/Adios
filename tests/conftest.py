import os
import sys

# Use a throwaway in-memory database so tests never touch predictor.db
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import app as app_module


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    with app_module.app.app_context():
        app_module.db.drop_all()
        app_module.db.create_all()
    return app_module.app.test_client()


@pytest.fixture
def logged_in_client(client):
    client.post("/register", data={"email": "tester@nis.kz",
                                   "password": "Kazakh2026",
                                   "confirm_password": "Kazakh2026"})
    client.post("/login", data={"email": "tester@nis.kz", "password": "Kazakh2026"})
    return client

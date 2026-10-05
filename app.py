import os
import pickle
import secrets
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///predictor.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Mail config (credentials come from environment variables only)
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.environ.get("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_USERNAME", "noreply@predictor.kz")

db = SQLAlchemy(app)
mail = Mail(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
        return self.reset_token


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    crime_type = db.Column(db.String(50))
    severity = db.Column(db.Integer)
    prior_convictions = db.Column(db.Integer)
    age = db.Column(db.Integer)
    employment_status = db.Column(db.String(50))
    has_dependents = db.Column(db.Integer)
    predicted_sentence = db.Column(db.Float)
    deviation_label = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


#model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentence_model.pkl")
model_data = None


def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        print("Model loaded.")
    else:
        print("WARNING: sentence_model.pkl not found. Run save_model.py first.")


def analyze_case(case_dict):
    if model_data is None:
        return None
    model = model_data["model"]
    knn = model_data["knn"]
    df = model_data["df"]
    preprocessor = model.named_steps["preprocessing"]

    new_case = pd.DataFrame([case_dict])
    predicted = model.predict(new_case)[0]

    new_processed = preprocessor.transform(new_case)
    distances, indices = knn.kneighbors(new_processed)
    similar_cases = df.iloc[indices[0]]
    avg_similar = similar_cases["sentence_length"].mean()
    deviation = abs(predicted - avg_similar)

    if deviation < 1:
        deviation_label = "LOW"
    elif deviation < 2:
        deviation_label = "MEDIUM"
    else:
        deviation_label = "HIGH"

    group_avg = df.groupby("has_dependents")["sentence_length"].mean()

    return {
        "predicted": round(predicted, 2),
        "similar_cases": similar_cases[["crime_type", "sentence_length"]].to_dict("records"),
        "deviation_label": deviation_label,
        "avg_no_dependents": round(group_avg.get(0, 0), 2),
        "avg_has_dependents": round(group_avg.get(1, 0), 2),
    }


# authentication decorator
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# vkladki

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("predict"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not email or not password:
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
        else:
            user = User(email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session["user_id"] = user.id
            session["user_email"] = user.email
            return redirect(url_for("predict"))
        else:
            flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()

        if user:
            token = user.generate_reset_token()
            db.session.commit()
            reset_url = url_for("reset_password", token=token, _external=True)

            try:
                msg = Message(
                    subject="Password Reset",
                    recipients=[email],
                    body=f"Reset your password here (expires in 1 hour):\n\n{reset_url}\n\nIf you did not request this, ignore this email."
                )
                mail.send(msg)
            except Exception as e:
                print(f"Mail error: {e}")

        # Always show same message to prevent email enumeration
        flash("If that email exists, a reset link has been sent.", "success")
        return redirect(url_for("login"))

    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()

    if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
        flash("Reset link is invalid or has expired.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        else:
            user.set_password(password)
            user.reset_token = None
            user.reset_token_expires = None
            db.session.commit()
            flash("Password updated. Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("reset_password.html", token=token)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    valid_crimes = ["theft", "fraud", "assault", "robbery"]
    valid_employment = ["employed", "unemployed", "student"]

    if request.method == "POST":
        crime_type = request.form.get("crime_type")
        severity = request.form.get("severity")
        prior_convictions = request.form.get("prior_convictions")
        age = request.form.get("age")
        employment_status = request.form.get("employment_status")
        has_dependents = request.form.get("has_dependents", "0")

        errors = []
        if crime_type not in valid_crimes:
            errors.append("Invalid crime type.")
        try:
            severity = int(severity)
            if not 1 <= severity <= 5:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Severity must be 1–5.")
        try:
            prior_convictions = int(prior_convictions)
            if not 0 <= prior_convictions <= 20:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Prior convictions must be 0–20.")
        try:
            age = int(age)
            if not 18 <= age <= 60:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Age must be 18–60.")
        if employment_status not in valid_employment:
            errors.append("Invalid employment status.")
        if has_dependents in ("0", "1"):
            has_dependents = int(has_dependents)
        else:
            errors.append("Has dependents must be Yes or No.")

        if errors:
            for e in errors:
                flash(e, "error")
        elif model_data is None:
            flash("Model not loaded. Please run save_model.py first.", "error")
        else:
            case = {
                "crime_type": crime_type,
                "severity": severity,
                "prior_convictions": prior_convictions,
                "age": age,
                "employment_status": employment_status,
                "has_dependents": has_dependents,
            }
            result = analyze_case(case)

            pred = Prediction(
                user_id=session["user_id"],
                crime_type=crime_type,
                severity=severity,
                prior_convictions=prior_convictions,
                age=age,
                employment_status=employment_status,
                has_dependents=has_dependents,
                predicted_sentence=result["predicted"],
                deviation_label=result["deviation_label"],
            )
            db.session.add(pred)
            db.session.commit()

    return render_template("predict.html",
                           result=result,
                           valid_crimes=valid_crimes,
                           valid_employment=valid_employment)


@app.route("/history")
@login_required
def history():
    preds = Prediction.query.filter_by(user_id=session["user_id"]) \
                            .order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=preds)


# init

with app.app_context():
    db.create_all()
    load_model()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

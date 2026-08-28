"""
Main Flask application for the Credit Scoring Prediction System.

Provides authentication, prediction, dashboard, and history features.
"""

import os
import json
import io
import csv
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    session, send_file, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from functools import wraps

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from predict import CreditRiskPredictor
from helper import read_json_file, generate_application_id, risk_badge

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    os.path.dirname(__file__), "credit.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Load predictor lazily
predictor = None


def get_predictor():
    """Load the credit risk predictor (cached)."""
    global predictor
    if predictor is None:
        try:
            predictor = CreditRiskPredictor()
        except Exception as e:
            print(f"⚠️ Predictor not loaded: {e}")
    return predictor


# ---------------------- Database Models ----------------------
class User(db.Model):
    """User account model."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), default="user")  # user or admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship("Prediction", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    """Prediction record model."""
    id = db.Column(db.Integer, primary_key=True)
    application_id = db.Column(db.String(30), unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    risk_level = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    probability_low = db.Column(db.Float)
    probability_medium = db.Column(db.Float)
    probability_high = db.Column(db.Float)
    input_data = db.Column(db.Text)  # JSON string of inputs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "application_id": self.application_id,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "probability_low": self.probability_low,
            "probability_medium": self.probability_medium,
            "probability_high": self.probability_high,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


# ---------------------- Auth Decorators ----------------------
def login_required(f):
    """Decorator to require login for a route."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json:
                return jsonify({"error": "Not authenticated"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator to require admin role for a route."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        user = db.session.get(User, session["user_id"])
        if user is None or user.role != "admin":
            if request.is_json:
                return jsonify({"error": "Admin access required"}), 403
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


# ---------------------- Page Routes ----------------------
@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            session["username"] = user.username
            session["role"] = user.role
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup page."""
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")

        if password != confirm:
            flash("Passwords do not match", "danger")
            return redirect(url_for("signup"))

        if User.query.filter_by(username=username).first():
            flash("Username already taken", "danger")
            return redirect(url_for("signup"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return redirect(url_for("signup"))

        user = User(username=username, email=email, role="user")
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    """Logout."""
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("index"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """Prediction form and processing."""
    if request.method == "POST":
        data = request.form.to_dict()
        data["Annual_Income"] = float(data.get("Annual_Income", 0))
        data["Monthly_Income"] = float(data.get("Monthly_Income", 0))
        data["Loan_Amount"] = float(data.get("Loan_Amount", 0))
        data["Debt"] = float(data.get("Debt", 0))
        data["Credit_Card_Debt"] = float(data.get("Credit_Card_Debt", 0))
        data["Savings"] = float(data.get("Savings", 0))
        data["Investments"] = float(data.get("Investments", 0))
        data["Payment_History"] = float(data.get("Payment_History", 0))
        data["Credit_Utilization"] = float(data.get("Credit_Utilization", 0))
        data["Bank_Balance"] = float(data.get("Bank_Balance", 0))
        data["Assets"] = float(data.get("Assets", 0))
        data["Existing_Loans"] = int(data.get("Existing_Loans", 0))
        data["Missed_Payments"] = int(data.get("Missed_Payments", 0))
        data["Number_of_Credit_Cards"] = int(data.get("Number_of_Credit_Cards", 0))
        data["Employment_Years"] = float(data.get("Employment_Years", 0))
        data["Age"] = int(data.get("Age", 0))
        data["Dependents"] = int(data.get("Dependents", 0))
        data["Previous_Defaults"] = int(data.get("Previous_Defaults", 0))
        data["Credit_History_Length_Years"] = float(data.get("Credit_History_Length_Years", 0))
        data["Loan_Term_Months"] = int(data.get("Loan_Term_Months", 0))
        data["Interest_Rate"] = float(data.get("Interest_Rate", 0))
        data["EMI"] = float(data.get("EMI", 0))

        pred = get_predictor()
        if pred is None:
            flash("Model not trained yet. Please run training first.", "danger")
            return redirect(url_for("predict"))

        result = pred.predict(data)

        # Save prediction to database
        user = db.session.get(User, session["user_id"])
        input_json = json.dumps(data)
        prob = result["probability"]
        record = Prediction(
            application_id=generate_application_id(),
            user_id=user.id,
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            probability_low=prob.get("Low Risk", 0),
            probability_medium=prob.get("Medium Risk", 0),
            probability_high=prob.get("High Risk", 0),
            input_data=input_json,
        )
        db.session.add(record)
        db.session.commit()

        result["application_id"] = record.application_id
        result["input_data"] = data
        return render_template("result.html", result=result)

    return render_template("prediction.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard with charts and history."""
    user = db.session.get(User, session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).order_by(
        Prediction.created_at.desc()
    ).all()

    # Stats
    total = len(predictions)
    dist = {"Low": 0, "Medium": 0, "High": 0}
    for p in predictions:
        if p.risk_level in dist:
            dist[p.risk_level] += 1

    # Model metadata
    meta_path = os.path.join(os.path.dirname(__file__), "model", "model_metadata.json")
    metadata = read_json_file(meta_path, {})
    model_accuracy = metadata.get("best_accuracy", 0)
    best_model = metadata.get("best_model", "N/A")
    results = metadata.get("results", {})
    feature_importance = get_predictor().get_feature_importance() if get_predictor() else None

    return render_template(
        "dashboard.html",
        total=total,
        dist=dist,
        predictions=predictions,
        model_accuracy=model_accuracy,
        best_model=best_model,
        results=results,
        feature_importance=feature_importance,
        current_user=user,
    )


@app.route("/admin")
@admin_required
def admin_dashboard():
    """Admin dashboard showing all predictions."""
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    users = User.query.all()
    total_users = len(users)
    total_preds = len(predictions)
    return render_template(
        "admin_dashboard.html",
        predictions=predictions,
        users=users,
        total_users=total_users,
        total_preds=total_preds,
    )


# ---------------------- API Routes ----------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions (JSON)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pred = get_predictor()
    if pred is None:
        return jsonify({"error": "Model not trained"}), 500

    try:
        result = pred.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
@login_required
def api_history():
    """API endpoint for prediction history."""
    user = db.session.get(User, session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).all()
    return jsonify([p.to_dict() for p in predictions])


@app.route("/api/train", methods=["POST"])
def api_train():
    """API endpoint to trigger training."""
    try:
        from train_model import train
        metadata = train()
        global predictor
        predictor = None  # reset cache
        return jsonify({"status": "success", "metadata": metadata})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/stats")
@login_required
def api_dashboard_stats():
    """API endpoint for dashboard statistics."""
    user = db.session.get(User, session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).all()
    dist = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
    for p in predictions:
        key = p.risk_level + " Risk" if p.risk_level != "Risk" else "Medium Risk"
        for k in dist:
            if p.risk_level in k:
                dist[k] += 1
                break
    return jsonify({"total": len(predictions), "distribution": dist})


# ---------------------- Export Routes ----------------------
@app.route("/history/export/csv")
@login_required
def export_csv():
    """Export prediction history as CSV."""
    user = db.session.get(User, session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Application ID", "Risk Level", "Confidence",
        "Probability (Low)", "Probability (Medium)", "Probability (High)", "Date"
    ])
    for p in predictions:
        writer.writerow([
            p.application_id, p.risk_level, p.confidence,
            p.probability_low, p.probability_medium, p.probability_high,
            p.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv",
    )


@app.route("/history/export/pdf")
@login_required
def export_pdf():
    """Export prediction history as PDF."""
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

    user = db.session.get(User, session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).all()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("<b>Credit Scoring Prediction History</b>", styles["Title"])
    elements.append(title)
    elements.append(Paragraph(f"User: {user.username} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    data = [["App ID", "Risk", "Confidence", "Prob Low", "Prob Med", "Prob High", "Date"]]
    for p in predictions:
        data.append([
            p.application_id, p.risk_level, f"{p.confidence}%",
            f"{p.probability_low}%", f"{p.probability_medium}%", f"{p.probability_high}%",
            p.created_at.strftime("%Y-%m-%d %H:%M")
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="prediction_history.pdf",
    )


# ---------------------- History Management ----------------------
@app.route("/history")
@login_required
def history():
    """Prediction history page."""
    user = db.session.get(User, session["user_id"])
    predictions = Prediction.query.filter_by(user_id=user.id).order_by(
        Prediction.created_at.desc()
    ).all()
    return render_template("history.html", predictions=predictions)


@app.route("/history/delete/<int:pred_id>", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    """Delete a single prediction."""
    pred = db.session.get(Prediction, pred_id)
    if pred and (pred.user_id == session["user_id"] or session.get("role") == "admin"):
        db.session.delete(pred)
        db.session.commit()
        flash("Prediction deleted", "success")
    else:
        flash("Not authorized", "danger")
    return redirect(request.referrer or url_for("history"))


# ---------------------- Error Handlers ----------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    return "Internal server error", 500


# ---------------------- Main Entry ----------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        # Create default admin if not exists
        if not User.query.filter_by(role="admin").first():
            admin = User(username="admin", email="admin@creditscoring.com", role="admin")
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()
            print("✅ Default admin created (username: admin, password: admin123)")

    print("🚀 Starting Credit Scoring Prediction System...")
    app.run(debug=True, host="0.0.0.0", port=5000)

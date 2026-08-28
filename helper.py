"""
Helper utilities for the Credit Scoring Prediction System.
"""

import os
import json
import random
import string
from datetime import datetime


def generate_application_id():
    """Generate a unique application ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"CS-{timestamp}-{rand}"


def serialize_datetime(obj):
    """Convert datetime objects to ISO strings for JSON serialization."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def json_response(data, status=200):
    """Create a JSON response wrapper."""
    from flask import jsonify
    return jsonify(data), status


def read_json_file(path, default=None):
    """Safely read a JSON file."""
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}


def write_json_file(path, data):
    """Safely write a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=serialize_datetime)


def format_currency(value):
    """Format a number as currency."""
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def risk_badge(risk_level):
    """Return a CSS class for risk level."""
    mapping = {
        "Low": "success",
        "Medium": "warning",
        "High": "danger",
    }
    return mapping.get(risk_level, "secondary")

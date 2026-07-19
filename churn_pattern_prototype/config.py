"""
config.py - Database connection and prototype-wide settings.

Loads DB credentials from environment variables or a local .env file
using python-dotenv. Never hardcodes credentials.

This module provides:
  - Database connection factory via get_db_connection()
  - Output directory path
  - Logging configuration
  - Feature category keyword mappings used for pattern auto-labeling

No ML parameters or numeric thresholds are defined here - those are
function parameters in their respective modules.
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the prototype directory (same folder as this file)
# ---------------------------------------------------------------------------
_PROTOTYPE_DIR = Path(__file__).resolve().parent
load_dotenv(_PROTOTYPE_DIR / ".env")

# ---------------------------------------------------------------------------
# Database credentials - read from environment variables only
# ---------------------------------------------------------------------------
ML_DB_HOST = os.getenv("ML_DB_HOST")
ML_DB_USER = os.getenv("ML_DB_USER")
ML_DB_PASSWORD = os.getenv("ML_DB_PASSWORD")
ML_DB_DATABASE = os.getenv("ML_DB_DATABASE")
ML_DB_PORT = int(os.getenv("ML_DB_PORT", "3306"))

# ---------------------------------------------------------------------------
# Prototype output directory - all CSV/JSON results are saved here
# ---------------------------------------------------------------------------
PROTOTYPE_OUTPUT_DIR = str(_PROTOTYPE_DIR / "output")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("nmf_prototype")

# ---------------------------------------------------------------------------
# Feature category keyword mappings - used by pattern_engine.auto_label_patterns
# to classify feature names into behavioral categories.
#
# Each key is a category label; each value is a list of substrings.
# If a feature name contains any of the substrings, it is assigned that category.
# Order matters: first match wins.
# ---------------------------------------------------------------------------
FEATURE_CATEGORY_KEYWORDS = {
    "engagement": [
        "meeting", "engagement", "activities", "notes", "days_since_last_meeting",
    ],
    "revenue": [
        "revenue", "purchase", "product_amount", "subscription", "active_subscriptions",
    ],
    "support": [
        "ticket", "sla", "reopen", "days_since_last_ticket",
    ],
    "risk": [
        "red_flag", "lost_opportunit", "flag",
    ],
    "staleness": [
        "days_since", "stale", "age_days",
    ],
    "account": [
        "key_account", "primary_contact", "customer_age",
    ],
}


def get_db_connection():
    """
    Create and return a new MySQL database connection using mysql-connector-python.

    Returns:
        mysql.connector.connection.MySQLConnection: An open database connection.

    Raises:
        SystemExit: If required environment variables are missing.
        mysql.connector.Error: If the connection cannot be established.
    """
    import mysql.connector

    missing = []
    if not ML_DB_HOST:
        missing.append("ML_DB_HOST")
    if not ML_DB_USER:
        missing.append("ML_DB_USER")
    if not ML_DB_DATABASE:
        missing.append("ML_DB_DATABASE")
    # ML_DB_PASSWORD can be empty string for passwordless local setups

    if missing:
        logger.error(
            "Missing required DB environment variables: %s. "
            "Please set them in the .env file.",
            ", ".join(missing),
        )
        sys.exit(1)

    connection = mysql.connector.connect(
        host=ML_DB_HOST,
        port=ML_DB_PORT,
        user=ML_DB_USER,
        password=ML_DB_PASSWORD or "",
        database=ML_DB_DATABASE,
        charset="utf8mb4",
        use_unicode=True,
        connection_timeout=30,
    )
    return connection


def validate_db_connection():
    """
    Test the database connection and print diagnostic information.

    Returns:
        bool: True if connection succeeded, False otherwise.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        logger.info(
            "[OK] Database connection successful: %s@%s:%s/%s",
            ML_DB_USER, ML_DB_HOST, ML_DB_PORT, ML_DB_DATABASE,
        )
        return True
    except Exception as exc:
        logger.error("[FAIL] Database connection failed: %s", exc)
        return False

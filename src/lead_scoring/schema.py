from __future__ import annotations

TARGET = "Completed Purchase"
ID_COLUMN = "Lead ID"
TIME_COLUMN = "Created At"

CSV_COLUMNS = [
    "Lead ID",
    "Created At",
    "Product Type",
    "Channel",
    "Device",
    "Partner",
    "City",
    "Insurance Company",
    "Payment Type",
    "Minutes Since Abandonment",
    "Days To Policy Expiry",
    "Price",
    "Discount Percent",
    "Has Previous Purchase",
    "Visited Offer Page",
    "Incoming Call Last 24h",
    "Sessions Last 7d",
    "Offer Views Last 7d",
    "Price Comparisons Last 7d",
    "Days Since Last Visit",
    "Expected Margin",
    "Completed Purchase",
]

DB_COLUMNS = [column.lower().replace(" ", "_") for column in CSV_COLUMNS]

NUMERIC_FEATURES = [
    "Minutes Since Abandonment",
    "Days To Policy Expiry",
    "Price",
    "Discount Percent",
    "Has Previous Purchase",
    "Visited Offer Page",
    "Incoming Call Last 24h",
    "Sessions Last 7d",
    "Offer Views Last 7d",
    "Price Comparisons Last 7d",
    "Days Since Last Visit",
    "Expected Margin",
    "Created Hour",
    "Created Day Of Week",
]

CATEGORICAL_FEATURES = [
    "Product Type",
    "Channel",
    "Device",
    "Partner",
    "City",
    "Insurance Company",
    "Payment Type",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BINARY_FEATURES = [
    "Has Previous Purchase",
    "Visited Offer Page",
    "Incoming Call Last 24h",
]

RANGES: dict[str, tuple[float, float]] = {
    "Minutes Since Abandonment": (0, 7 * 24 * 60),
    "Days To Policy Expiry": (-365, 365),
    "Price": (0, 1_000_000_000),
    "Discount Percent": (0, 100),
    "Sessions Last 7d": (0, 10_000),
    "Offer Views Last 7d": (0, 10_000),
    "Price Comparisons Last 7d": (0, 10_000),
    "Days Since Last Visit": (0, 3650),
    "Expected Margin": (0, 1_000_000_000),
}

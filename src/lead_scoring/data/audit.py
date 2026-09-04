from __future__ import annotations

import pandas as pd


def feature_availability_audit() -> pd.DataFrame:
    """Return the conservative, operational feature contract."""
    rows = [
        (
            "Lead ID",
            "Available at scoring time",
            "Identifier or operational metadata",
            "Exclude",
            "High-cardinality identifier; used only for lineage and stable ranking.",
        ),
        (
            "Created At",
            "Available at scoring time",
            "Questionable and requiring an assumption",
            "Derived only",
            "Lead creation time is assumed immutable; only hour/day-of-week are used, "
            "not absolute date.",
        ),
        (
            "Product Type",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Insurance product requested before abandonment.",
        ),
        (
            "Channel",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Acquisition attribution is assumed fixed at lead creation.",
        ),
        (
            "Device",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Observed before abandonment.",
        ),
        (
            "Partner",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Lead source is assumed known before telesales eligibility.",
        ),
        (
            "City",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "User-entered geography; missingness is explicitly modeled.",
        ),
        (
            "Insurance Company",
            "Available at scoring time",
            "Questionable and requiring an assumption",
            "Include",
            "Assumed to be the pre-purchase selected/requested insurer, not the "
            "completed-order insurer.",
        ),
        (
            "Payment Type",
            "Available at scoring time",
            "Questionable and requiring an assumption",
            "Include",
            "Assumed selected during quote flow before abandonment; event lineage must be "
            "verified in production.",
        ),
        (
            "Minutes Since Abandonment",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Defines eligibility and time elapsed at the scoring snapshot.",
        ),
        (
            "Days To Policy Expiry",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Existing policy timing is known during quote flow; negative values mean "
            "already expired.",
        ),
        (
            "Price",
            "Available at scoring time",
            "Questionable and requiring an assumption",
            "Include",
            "Assumed quoted price, not settled purchase price.",
        ),
        (
            "Discount Percent",
            "Available at scoring time",
            "Questionable and requiring an assumption",
            "Include",
            "Assumed offered discount visible before abandonment, not a post-outcome concession.",
        ),
        (
            "Has Previous Purchase",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Historical purchase state predates the current lead.",
        ),
        (
            "Visited Offer Page",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Funnel event occurs before the defined post-abandonment scoring point.",
        ),
        (
            "Incoming Call Last 24h",
            "Available at scoring time",
            "Questionable and requiring an assumption",
            "Include",
            "Only inbound calls before scoring are allowed; outbound post-score activity "
            "must never populate it.",
        ),
        (
            "Sessions Last 7d",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Trailing window must be event-time bounded at scoring.",
        ),
        (
            "Offer Views Last 7d",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Trailing funnel events occur before scoring.",
        ),
        (
            "Price Comparisons Last 7d",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Trailing comparisons occur before scoring.",
        ),
        (
            "Days Since Last Visit",
            "Available at scoring time",
            "Valid candidate feature",
            "Include",
            "Snapshot feature must be recomputed as of scoring time.",
        ),
        (
            "Expected Margin",
            "Available at scoring time",
            "Questionable and requiring an assumption",
            "Include",
            "Expected pre-purchase economics, not realized margin; verify calculation lineage.",
        ),
        (
            "Completed Purchase",
            "Unavailable or created after the outcome",
            "Direct target leakage",
            "Exclude",
            "Outcome label only; never passed to preprocessing or scoring.",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["column", "availability", "classification", "model_use", "rationale"]
    )

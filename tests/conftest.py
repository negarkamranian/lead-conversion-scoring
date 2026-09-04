from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_scoring.schema import CSV_COLUMNS


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = []
    for i in range(60):
        rows.append(
            {
                "Lead ID": f"L{i:06d}",
                "Created At": f"2026-01-{1 + i // 24:02d} {i % 24:02d}:00:00",
                "Product Type": "thirdparty" if i % 2 else "carbody",
                "Channel": ["SEO", "Paid", "CRM"][i % 3],
                "Device": "mobile" if i % 2 else "desktop",
                "Partner": "direct",
                "City": "Tehran" if i != 5 else "",
                "Insurance Company": "Iran",
                "Payment Type": "cash",
                "Minutes Since Abandonment": str(i + 1),
                "Days To Policy Expiry": str(i % 30 - 5),
                "Price": "" if i == 4 else str(5_000_000 + i * 1000),
                "Discount Percent": str(i % 10),
                "Has Previous Purchase": str(i % 2),
                "Visited Offer Page": "1",
                "Incoming Call Last 24h": "0",
                "Sessions Last 7d": str(1 + i % 5),
                "Offer Views Last 7d": str(i % 4),
                "Price Comparisons Last 7d": str(i % 3),
                "Days Since Last Visit": str(i % 12),
                "Expected Margin": str(100_000 + i * 100),
                "Completed Purchase": "1" if i % 7 == 0 else "0",
            }
        )
    return pd.DataFrame(rows, columns=CSV_COLUMNS).replace({np.nan: ""})

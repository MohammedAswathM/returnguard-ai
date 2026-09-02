from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from returnguard.data.uci import build_uci_foundation


def test_uci_foundation_preserves_cancellations_as_non_label_events(tmp_path: Path) -> None:
    workbook = tmp_path / "online_retail_ii.xlsx"
    rows = []
    for index in range(120):
        rows.append({
            "Invoice": f"{50000 + index}", "StockCode": f"S{index % 8}",
            "Description": "fixture product", "Quantity": 1 + index % 3,
            "InvoiceDate": pd.Timestamp("2010-01-01") + pd.Timedelta(days=index),
            "Price": 5.0 + index % 7, "Customer ID": 1000 + index % 20, "Country": "United Kingdom",
        })
    rows.append({**rows[0], "Invoice": "C50000", "Quantity": -1})
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Year 2009-2010", index=False)
    archive = tmp_path / "online_retail_ii.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        bundle.write(workbook, workbook.name)
    customers, orders, payments, order_items, events, notes = build_uci_foundation(
        archive, 100, 7, 100.0
    )
    assert len(orders) == len(payments) == 100
    assert len(customers) > 0
    assert len(order_items) >= 100
    assert len(events) == 1
    assert notes["uci_cancellation_transaction_rows"] == 1
    assert notes["uci_cancellations_used_as_labels"] is False
    assert "is_refund_abuse_simulated" not in orders

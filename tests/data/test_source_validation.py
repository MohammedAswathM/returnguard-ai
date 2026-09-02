from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from returnguard.data.validation import EXPECTED_COLUMNS, validate_uci_archive


def test_source_validation_exports_required_statistics(tmp_path: Path) -> None:
    workbook = tmp_path / "online_retail_ii.xlsx"
    frame = pd.DataFrame([{
        "Invoice": "50000", "StockCode": "S1", "Description": "product", "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2010-01-01"), "Price": 5.0,
        "Customer ID": 1000, "Country": "United Kingdom",
    }], columns=EXPECTED_COLUMNS)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        frame.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        bundle.write(workbook, workbook.name)
    report_path = tmp_path / "validation.json"
    report = validate_uci_archive(archive, report_path)
    assert report["status"] == "VALIDATED_SCHEMA_COMPATIBLE"
    assert report["official_identity_checks"]["all_match"] is False
    assert report["totals"]["rows"] == 2
    assert len(report["archive"]["sha256"]) == 64
    assert report["source"]["license"] == "CC BY 4.0"
    assert report_path.exists()

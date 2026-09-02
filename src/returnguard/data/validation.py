from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from returnguard.data.download import UCI_DOI, UCI_LICENSE, UCI_URL
from returnguard.data.fingerprints import file_sha256, object_sha256

EXPECTED_SHEETS = ("Year 2009-2010", "Year 2010-2011")
EXPECTED_COLUMNS = (
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
    "Customer ID", "Country",
)
OFFICIAL_ARCHIVE_BYTES = 45_622_418
OFFICIAL_ARCHIVE_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
OFFICIAL_TOTAL_ROWS = 1_067_371


def validate_uci_archive(archive: Path, report_path: Path | None = None) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(f"UCI archive not found: {archive}")
    with ZipFile(archive) as bundle:
        spreadsheet_names = sorted(
            name for name in bundle.namelist() if name.lower().endswith((".xlsx", ".xls"))
        )
        if len(spreadsheet_names) != 1:
            raise ValueError("official archive must contain exactly one spreadsheet")
        member = spreadsheet_names[0]
        member_size = bundle.getinfo(member).file_size
        with bundle.open(member) as workbook:
            excel = pd.ExcelFile(workbook)
            sheet_names = tuple(excel.sheet_names)
            if sheet_names != EXPECTED_SHEETS:
                raise ValueError(f"unexpected worksheet names: {sheet_names}")
            worksheets: list[dict[str, Any]] = []
            for sheet_name in sheet_names:
                frame = pd.read_excel(excel, sheet_name=sheet_name)
                columns = tuple(str(column) for column in frame.columns)
                if columns != EXPECTED_COLUMNS:
                    raise ValueError(f"unexpected columns in {sheet_name}: {columns}")
                invoice = frame["Invoice"].astype(str)
                dates = pd.to_datetime(frame["InvoiceDate"], errors="coerce", utc=True)
                worksheets.append({
                    "name": sheet_name,
                    "rows": int(len(frame)),
                    "columns": list(columns),
                    "timestamp_min": dates.min().isoformat(),
                    "timestamp_max": dates.max().isoformat(),
                    "missing_by_column": {
                        column: int(value) for column, value in frame.isna().sum().items()
                    },
                    "exact_duplicate_rows": int(frame.duplicated().sum()),
                    "unique_customers": int(frame["Customer ID"].nunique(dropna=True)),
                    "missing_customer_rows": int(frame["Customer ID"].isna().sum()),
                    "cancellation_rows": int(invoice.str.startswith("C").sum()),
                    "nonpositive_quantity_rows": int((frame["Quantity"] <= 0).sum()),
                    "nonpositive_price_rows": int((frame["Price"] <= 0).sum()),
                })
    totals = {
        field: sum(int(sheet[field]) for sheet in worksheets)
        for field in (
            "rows", "exact_duplicate_rows", "missing_customer_rows", "cancellation_rows",
            "nonpositive_quantity_rows", "nonpositive_price_rows",
        )
    }
    archive_sha256 = file_sha256(archive)
    official_identity = (
        archive.stat().st_size == OFFICIAL_ARCHIVE_BYTES
        and archive_sha256 == OFFICIAL_ARCHIVE_SHA256
        and totals["rows"] == OFFICIAL_TOTAL_ROWS
    )
    report: dict[str, Any] = {
        "validation_schema_version": "1.0",
        "status": (
            "VALIDATED_OFFICIAL_SOURCE" if official_identity else "VALIDATED_SCHEMA_COMPATIBLE"
        ),
        "source": {"url": UCI_URL, "doi": UCI_DOI, "license": UCI_LICENSE},
        "archive": {
            "path": str(archive), "bytes": archive.stat().st_size,
            "sha256": archive_sha256, "members": [member],
            "workbook_uncompressed_bytes": member_size,
        },
        "official_identity_checks": {
            "expected_bytes": OFFICIAL_ARCHIVE_BYTES,
            "expected_sha256": OFFICIAL_ARCHIVE_SHA256,
            "expected_rows": OFFICIAL_TOTAL_ROWS,
            "all_match": official_identity,
        },
        "worksheets": worksheets,
        "totals": totals,
    }
    report["validation_fingerprint_sha256"] = object_sha256(report)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

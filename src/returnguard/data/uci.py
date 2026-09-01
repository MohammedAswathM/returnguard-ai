from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd


def _pseudonym(prefix: str, value: object) -> str:
    digest = hashlib.sha256(str(value).encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _read_workbook(archive: Path) -> pd.DataFrame:
    if not archive.exists():
        raise FileNotFoundError(f"UCI archive not found: {archive}")
    if archive.suffix.lower() == ".zip":
        with ZipFile(archive) as bundle:
            names = [name for name in bundle.namelist() if name.lower().endswith((".xlsx", ".xls"))]
            if len(names) != 1:
                raise ValueError("expected exactly one spreadsheet in UCI archive")
            workbook: Any = BytesIO(bundle.read(names[0]))
            sheets = pd.read_excel(workbook, sheet_name=None)
    else:
        sheets = pd.read_excel(archive, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def build_uci_foundation(
    archive: Path, n_orders: int, seed: int, gbp_to_inr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create canonical orders from UCI transactions; cancellations remain non-label events."""
    raw = _read_workbook(archive)
    raw.columns = [str(column).strip().lower().replace(" ", "_") for column in raw.columns]
    aliases = {"invoice": "invoice", "invoiceno": "invoice", "customer_id": "customer_id",
               "invoicedate": "invoice_date", "invoice_date": "invoice_date",
               "unitprice": "price", "price": "price", "stockcode": "stock_code",
               "stock_code": "stock_code", "quantity": "quantity", "country": "country"}
    raw = raw.rename(columns={column: aliases.get(column, column) for column in raw.columns})
    required = {"invoice", "customer_id", "invoice_date", "price", "quantity", "stock_code", "country"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"UCI workbook missing columns: {sorted(missing)}")
    raw = raw.dropna(subset=["invoice", "customer_id", "invoice_date", "price", "quantity"])
    raw["invoice"] = raw["invoice"].astype(str)
    cancellations = raw["invoice"].str.startswith("C")
    cancellation_event_count = int(cancellations.sum())
    clean = raw.loc[~cancellations & (raw["quantity"] > 0) & (raw["price"] > 0)].copy()
    clean["invoice_date"] = pd.to_datetime(clean["invoice_date"], utc=True)
    clean["line_value_gbp"] = clean["quantity"] * clean["price"]
    grouped = clean.groupby(["invoice", "customer_id"], as_index=False).agg(
        ordered_at=("invoice_date", "min"), gross_gbp=("line_value_gbp", "sum"),
        item_count=("quantity", "sum"), country=("country", "first"),
        stock_code=("stock_code", "first"),
    )
    if len(grouped) < n_orders:
        raise ValueError(f"UCI foundation has {len(grouped)} eligible orders, fewer than {n_orders}")
    sampled = grouped.sample(n=n_orders, random_state=seed).sort_values("ordered_at").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    customer_map = {
        value: _pseudonym("cus", value) for value in sorted(sampled["customer_id"].unique(), key=str)
    }
    order_rows: list[dict[str, Any]] = []
    payment_rows: list[dict[str, Any]] = []
    for index, row in sampled.iterrows():
        ordered = row["ordered_at"].to_pydatetime()
        delivered = ordered + pd.Timedelta(days=int(rng.integers(2, 9)))
        customer_id = customer_map[row["customer_id"]]
        order_id = f"ord_{index:07d}"
        gross_paise = max(1, round(float(row["gross_gbp"]) * gbp_to_inr * 100))
        category = f"uci_product_group_{hashlib.sha256(str(row['stock_code']).encode()).hexdigest()[:2]}"
        order_rows.append({
            "order_id": order_id, "customer_id": customer_id, "razorpay_order_id": "",
            "ordered_at": ordered.isoformat().replace("+00:00", "Z"),
            "delivered_at": delivered.isoformat().replace("+00:00", "Z"), "currency": "INR",
            "gross_amount_paise": gross_paise, "item_count": max(1, int(row["item_count"])),
            "category": category, "shipping_address_id": _pseudonym("addr", row["customer_id"]),
            "device_id": _pseudonym("dev", row["customer_id"]),
        })
        payment_rows.append({
            "payment_id": f"pay_{index:07d}", "razorpay_payment_id": "", "order_id": order_id,
            "paid_amount_paise": gross_paise, "status": "captured",
            "paid_at": (ordered + pd.Timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
        })
    orders = pd.DataFrame(order_rows)
    payments = pd.DataFrame(payment_rows)
    countries = sampled.groupby("customer_id")["country"].first()
    first_orders = sampled.groupby("customer_id")["ordered_at"].min()
    customers = pd.DataFrame({
        "customer_id": [customer_map[value] for value in countries.index],
        "created_at": [
            (first_orders[value] - pd.Timedelta(days=1)).isoformat().replace("+00:00", "Z")
            for value in countries.index
        ],
        "country": [str(countries[value]) for value in countries.index],
        "account_status": "active",
    })
    notes = {
        "uci_cancellation_transaction_rows": cancellation_event_count,
        "uci_cancellations_used_as_labels": False,
        "simulated_fx_assumption_gbp_to_inr": gbp_to_inr,
    }
    return customers, orders, payments, notes

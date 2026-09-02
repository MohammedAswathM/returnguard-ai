from typing import Any


def field_provenance(
    table_columns: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    explicit: dict[str, dict[str, dict[str, Any]]] = {
        "customers": {
            "customer_id": {
                "kind": "derived", "source": "Customer ID or Invoice when customer is missing",
                "transform": "SHA-256 pseudonym; invoice-scoped guest for missing customer",
            },
            "created_at": {
                "kind": "derived", "source": "first InvoiceDate",
                "transform": "first source event minus one day",
            },
            "country": {"kind": "source", "source": "Country"},
            "account_status": {"kind": "simulated", "source": None},
        },
        "orders": {
            "order_id": {"kind": "derived", "source": "Invoice"},
            "customer_id": {"kind": "derived", "source": "Customer ID"},
            "ordered_at": {"kind": "source", "source": "InvoiceDate"},
            "delivered_at": {"kind": "simulated", "source": None},
            "gross_amount_paise": {
                "kind": "derived", "source": "Quantity, Price",
                "transform": "configured GBP/INR assumption",
            },
            "item_count": {"kind": "derived", "source": "Quantity"},
            "category": {"kind": "derived", "source": "StockCode", "transform": "stable coarse product group"},
            "shipping_address_id": {"kind": "simulated", "source": None},
            "device_id": {"kind": "simulated", "source": None},
            "razorpay_order_id": {"kind": "simulated", "source": None},
        },
        "payments": {
            "payment_id": {"kind": "simulated", "source": None},
            "paid_amount_paise": {"kind": "derived", "source": "Quantity, Price"},
            "status": {"kind": "simulated", "source": None},
            "paid_at": {"kind": "simulated", "source": None},
            "razorpay_payment_id": {"kind": "simulated", "source": None},
        },
        "refund_requests": {
            "requested_at": {"kind": "simulated", "source": None},
            "reason_code": {"kind": "simulated", "source": None},
            "requested_amount_paise": {"kind": "simulated", "source": None},
            "requested_quantity": {"kind": "simulated", "source": None},
            "returnless_requested": {"kind": "simulated", "source": None},
            "evidence_provided": {"kind": "simulated", "source": None},
            "is_refund_abuse_simulated": {"kind": "simulated_label", "source": None},
            "simulation_scenario": {"kind": "generator_metadata", "feature_eligible": False},
            "outcome_available_at": {"kind": "simulated_outcome", "feature_eligible": False},
        },
        "verification_events": {
            "requested_at": {"kind": "simulated", "source": None},
            "completed_at": {"kind": "simulated", "source": None},
            "result": {"kind": "simulated_outcome", "feature_eligible": False},
            "result_codes": {"kind": "simulated_outcome", "feature_eligible": False},
        },
    }
    result: dict[str, dict[str, Any]] = {}
    for table, columns in table_columns.items():
        result[table] = {}
        for column in columns:
            details = explicit.get(table, {}).get(column)
            if details is None:
                if column in {"partition"}:
                    details = {"kind": "derived", "source": "requested_at"}
                elif column.endswith("_id") and table in {"orders", "order_items", "transaction_events"}:
                    details = {"kind": "derived", "source": "UCI transaction structure"}
                elif table in {"orders", "order_items", "transaction_events"}:
                    details = {"kind": "derived", "source": "UCI transaction structure"}
                else:
                    details = {"kind": "simulated", "source": None}
            result[table][column] = details
    return result

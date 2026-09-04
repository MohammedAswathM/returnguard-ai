from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
import streamlit as st

if TYPE_CHECKING:
    import pandas as pd

RESULTS_PATH = Path(os.getenv(
    "RETURNGUARD_RESULTS", "artifacts/v2/final_results/results.lock.json"
))
CORRECTION_PATH = Path(os.getenv(
    "RETURNGUARD_METRIC_CORRECTION", "results.v2.metric_integrity.v2.0.1.json"
))
DATABASE_PATH = Path(os.getenv("RETURNGUARD_DB", "artifacts/returnguard.db"))
API_URL = os.getenv("RETURNGUARD_API_URL", "http://127.0.0.1:8000")
DEMO_PAYLOAD_PATH = Path("artifacts/dashboard/demo_case_payload.json")


def load_results() -> dict[str, Any]:
    if not RESULTS_PATH.is_file():
        return {}
    return cast(dict[str, Any], json.loads(RESULTS_PATH.read_text(encoding="utf-8")))


def load_correction() -> dict[str, Any]:
    if not CORRECTION_PATH.is_file():
        return {}
    return cast(dict[str, Any], json.loads(CORRECTION_PATH.read_text(encoding="utf-8")))


def query(sql: str, parameters: tuple[Any, ...] = ()) -> pd.DataFrame:
    import pandas as pd

    if not DATABASE_PATH.is_file():
        return pd.DataFrame()
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(sql, connection, params=parameters)


def api_post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=20.0)
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


@st.cache_data
def demo_case_payload() -> dict[str, Any]:
    if DEMO_PAYLOAD_PATH.is_file():
        return cast(
            dict[str, Any], json.loads(DEMO_PAYLOAD_PATH.read_text(encoding="utf-8"))
        )
    return build_demo_case_payload()


def build_demo_case_payload() -> dict[str, Any]:
    import pandas as pd

    from returnguard.features.engine import build_point_in_time_features
    from returnguard.features.registry import load_feature_registry

    registry = load_feature_registry(Path("configs/features.yaml"))
    features = build_point_in_time_features(Path("artifacts/data_uci"), registry).set_index(
        "refund_request_id"
    )
    policy = pd.read_parquet("artifacts/frozen_policy/policy_selection_cases.parquet")
    candidates = policy.loc[
        (policy["stage_a_action"] == "VERIFY")
        & (policy["verification_result"] == "consistent")
        & (policy["adaptive_final_action"] == "AUTO_APPROVE")
    ]
    if candidates.empty:
        raise RuntimeError("frozen policy has no rescue candidate")
    row = cast(pd.Series, features.loc[str(candidates.iloc[0]["refund_request_id"])])
    return {
        "merchant_id": "demo-merchant",
        "customer_id": str(row["customer_id"]),
        "payment_id": f"demo-payment-{row.name}",
        "order_id": f"demo-order-{row.name}",
        "currency": "INR",
        "requested_amount_paise": int(row["requested_amount_paise"]),
        "payment_snapshot": {
            "merchant_id": "demo-merchant", "payment_id": f"demo-payment-{row.name}",
            "razorpay_payment_id": None, "order_id": f"demo-order-{row.name}",
            "currency": "INR", "captured_amount_paise": int(row["requested_amount_paise"]),
            "amount_refunded_paise": 0,
            "refundable_balance_paise": int(row["requested_amount_paise"]),
            "payment_status": "captured", "captured_at": None,
            "snapshot_as_of": datetime.now(UTC).isoformat(),
            "source": "LABELLED_DEMO_FIXTURE",
        },
        "features": {
            spec.name: row[spec.name].item() if hasattr(row[spec.name], "item") else row[spec.name]
            for spec in registry
        },
    }


def portfolio(results: dict[str, Any], correction: dict[str, Any]) -> None:
    st.header("Refund portfolio")
    if not results:
        st.warning("Locked evaluation artifact is unavailable.")
        return
    metrics = results["classifier"]
    if not correction:
        st.error("The authoritative metric-integrity correction is unavailable.")
        return
    adaptive = correction["corrections"]["policy_cost"]["comparison"]["adaptive_verification"]
    columns = st.columns(4)
    columns[0].metric("Requests", f"{results['support']:,}")
    columns[1].metric("Simulated abuse prevalence", f"{results['prevalence']:.2%}")
    columns[2].metric("Reviews / 1,000", f"{adaptive['manual_reviews_per_1000']:.2f}")
    legitimate_support = results["support"] - metrics["tp"] - metrics["fn"]
    rescue_count = round(adaptive["legitimate_rescue_rate"] * legitimate_support)
    columns[3].metric("Legitimate rescues", f"{rescue_count:,}")
    columns = st.columns(4)
    columns[0].metric("Raw PR-AUC", f"{metrics['raw_average_precision']:.3f}")
    columns[1].metric(
        "Simulated abuse-amount intervention",
        f"{adaptive['abuse_amount_intervention_recall']:.1%}",
    )
    columns[2].metric(
        "Initial legitimate challenge",
        f"{adaptive['initial_legitimate_challenge_rate']:.1%}",
    )
    columns[3].metric(
        "Challenged legitimate rescued",
        f"{adaptive['challenged_legitimate_rescue_rate']:.1%}",
    )
    st.caption(
        "V2 evaluates technically valid ambiguous claims separately from deterministic payment "
        "integrity. Labels, verification outcomes, and policy costs are simulated."
    )


def review_queue() -> None:
    st.header("Review queue")
    frame = query("""
        SELECT refund_request_id AS request, requested_amount_paise AS amount_paise,
               state AS status, updated_at AS updated
        FROM refund_cases WHERE state IN ('MANUAL_REVIEW', 'HELD', 'VERIFIED_INCONSISTENT')
        ORDER BY updated_at
    """)
    if frame.empty:
        st.info("No cases currently require operator review.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def case_detail() -> None:
    st.header("Case detail")
    cases = query("SELECT refund_request_id, state FROM refund_cases ORDER BY updated_at DESC")
    if cases.empty:
        st.info("No backend cases are available.")
        return
    request_id = st.selectbox("Refund request", cases["refund_request_id"].tolist())
    case = query("SELECT * FROM refund_cases WHERE refund_request_id = ?", (request_id,))
    decisions = query(
        "SELECT stage, probability, action, reason_codes_json, created_at FROM decisions "
        "WHERE refund_request_id = ? ORDER BY created_at", (request_id,),
    )
    audit = query(
        "SELECT sequence, event_type, before_state, after_state, reason, created_at FROM audit_events "
        "WHERE refund_request_id = ? ORDER BY sequence", (request_id,),
    )
    left, right = st.columns(2)
    left.subheader("Request")
    left.dataframe(case[["refund_request_id", "customer_id", "requested_amount_paise", "state"]])
    right.subheader("Immutable decisions")
    right.dataframe(decisions, hide_index=True)
    st.subheader("Append-only timeline")
    st.dataframe(audit, use_container_width=True, hide_index=True)


def evidence(results: dict[str, Any], correction: dict[str, Any]) -> None:
    import pandas as pd

    st.header("Held-out evidence")
    if not results:
        st.warning("Locked evaluation artifact is unavailable.")
        return
    metric = results["classifier"]
    matrix = pd.DataFrame(
        [[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]],
        index=["Actual legitimate", "Actual simulated abuse"],
        columns=["Predicted clear", "Predicted risk"],
    )
    st.subheader("Confusion counts")
    st.dataframe(matrix)
    report = json.loads(Path("evidence/v2/final_report.json").read_text(encoding="utf-8"))
    bins = pd.DataFrame(report["classifier"]["reliability_bins"])
    st.subheader("Reliability with bin support")
    st.line_chart(bins.set_index("mean_probability")["observed_prevalence"])
    st.dataframe(bins[["bin", "support", "mean_probability", "observed_prevalence"]], hide_index=True)
    st.subheader("Corrected confidence intervals")
    st.json(correction["corrections"]["bootstrap_intervals"]["values"])
    st.warning(
        "Transaction distributions are derived from UCI Online Retail II. Abuse labels and "
        "operational verification fields are simulated; these are not production-performance claims."
    )
    if correction:
        st.subheader("Frozen policy comparison")
        rows = []
        for name, value in correction["corrections"]["policy_cost"]["comparison"].items():
            rows.append({
                "policy": name,
                "modeled_cost_inr": value["modeled_cost"]["total_modeled_cost_paise"] / 100,
                "terminal_legitimate_intervention": value["terminal_legitimate_intervention_rate"],
                "legitimate_return_first": value["legitimate_return_first_burden"],
                "reviews_per_1000": value["manual_reviews_per_1000"],
                "abuse_case_intervention": value["abuse_case_intervention_recall"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Modeled simulation cost under frozen assumptions; not realized savings.")


def demo() -> None:
    st.header("Golden workflows")
    st.caption("The configured gateway state is returned by the backend and is never inferred here.")
    rescue_tab, inconsistent_tab, unavailable_tab = st.tabs([
        "Legitimate rescue", "Inconsistent evidence", "Safe failure",
    ])
    with rescue_tab:
        request_id = "demo-legitimate-rescue"
        if st.button("1. Create and score rescue case", type="primary"):
            payload = demo_case_payload()
            try:
                api_post(f"/api/v1/refund-requests/{request_id}", payload)
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 409:
                    raise
            st.session_state["demo_score"] = api_post(
                f"/api/v1/refund-requests/{request_id}/score"
            )
        if "demo_score" in st.session_state:
            st.json(st.session_state["demo_score"])
            if st.button("2. Verify consistent evidence and execute approved refund"):
                payload = demo_case_payload()
                verification = api_post(f"/api/v1/refund-requests/{request_id}/verifications")
                decision = api_post(f"/api/v1/verifications/{verification['verification_id']}/complete", {
                    "requested_amount_paise": payload["requested_amount_paise"],
                    "refundable_balance_paise": payload["requested_amount_paise"],
                    "requested_quantity": 1, "eligible_quantity": 1, "delivered": True,
                    "reason_code": "damaged", "evidence_token": "demo-consistent",
                })
                execution = api_post(f"/api/v1/refund-requests/{request_id}/execute")
                st.success("Consistent evidence reduced the posterior and restored approval.")
                st.json({"posterior_decision": decision, "execution": execution})
    with inconsistent_tab:
        request_id = "demo-inconsistent-evidence"
        if st.button("1. Create and score inconsistent case"):
            payload = demo_case_payload()
            try:
                api_post(f"/api/v1/refund-requests/{request_id}", payload)
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 409:
                    raise
            st.session_state["inconsistent_score"] = api_post(
                f"/api/v1/refund-requests/{request_id}/score"
            )
        if "inconsistent_score" in st.session_state and st.button(
            "2. Submit structured payment mismatch"
        ):
            payload = demo_case_payload()
            verification = api_post(f"/api/v1/refund-requests/{request_id}/verifications")
            decision = api_post(f"/api/v1/verifications/{verification['verification_id']}/complete", {
                "payment_matches_order": False,
                "requested_amount_paise": payload["requested_amount_paise"],
                "refundable_balance_paise": payload["requested_amount_paise"],
                "requested_quantity": 1, "eligible_quantity": 1, "delivered": True,
                "reason_code": "damaged", "evidence_token": "demo-consistent",
            })
            st.warning("The neutral inconsistency result requires named human review; no refund ran.")
            st.json(decision)
    with unavailable_tab:
        request_id = "demo-verifier-unavailable"
        if st.button("1. Create and score safe-failure case"):
            payload = demo_case_payload()
            try:
                api_post(f"/api/v1/refund-requests/{request_id}", payload)
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 409:
                    raise
            st.session_state["unavailable_score"] = api_post(
                f"/api/v1/refund-requests/{request_id}/score"
            )
        if "unavailable_score" in st.session_state and st.button("2. Disable verifier"):
            payload = demo_case_payload()
            verification = api_post(f"/api/v1/refund-requests/{request_id}/verifications")
            decision = api_post(f"/api/v1/verifications/{verification['verification_id']}/complete", {
                "verifier_available": False,
                "requested_amount_paise": payload["requested_amount_paise"],
                "refundable_balance_paise": payload["requested_amount_paise"],
                "requested_quantity": 1, "eligible_quantity": 1, "delivered": True,
                "reason_code": "damaged", "evidence_token": "demo-consistent",
            })
            st.warning("Verifier failure remained inconclusive and no autonomous adverse action ran.")
            st.json(decision)


def main() -> None:
    st.set_page_config(page_title="ReturnGuard", layout="wide")
    st.title("ReturnGuard")
    st.caption(
        "Evidence panels: frozen v2 offline benchmark | Golden workflows: v1 operational "
        "demonstration bundle | Razorpay evidence: execution-layer validation"
    )
    results = load_results()
    correction = load_correction()
    page = st.sidebar.radio(
        "View", ["Portfolio", "Review queue", "Case detail", "Evidence", "Golden workflows"]
    )
    if page == "Portfolio":
        portfolio(results, correction)
    elif page == "Review queue":
        review_queue()
    elif page == "Case detail":
        case_detail()
    elif page == "Evidence":
        evidence(results, correction)
    else:
        demo()


if __name__ == "__main__":
    main()

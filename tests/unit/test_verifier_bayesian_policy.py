import numpy as np
import pandas as pd

from returnguard.config import PolicyConfig
from returnguard.domain.enums import (
    PaymentStatus,
    RecommendedAction,
    RefundReason,
    VerificationResult,
)
from returnguard.policy.engine import choose_action
from returnguard.policy.selection import PolicyContract, apply_contract
from returnguard.verification.bayesian import bayesian_update, estimate_likelihoods
from returnguard.verification.order_integrity import IntegrityCheck, verify_order_integrity


def base_check(**changes: object) -> IntegrityCheck:
    values = {
        "order_exists": True, "payment_exists": True, "payment_matches_order": True,
        "payment_status": PaymentStatus.CAPTURED, "requested_amount_paise": 5000,
        "refundable_balance_paise": 10000, "requested_quantity": 1, "eligible_quantity": 2,
        "delivered": True, "reason_code": RefundReason.DAMAGED,
        "evidence_token": "demo-ok", "allowed_evidence_tokens": frozenset({"demo-ok"}),
    }
    values.update(changes)
    return IntegrityCheck(**values)  # type: ignore[arg-type]


def test_verifier_is_deterministic_and_missing_is_inconclusive() -> None:
    assert verify_order_integrity(base_check()).result == VerificationResult.CONSISTENT
    mismatch = verify_order_integrity(base_check(requested_amount_paise=10001))
    assert mismatch.result == VerificationResult.INCONSISTENT
    assert verify_order_integrity(base_check(evidence_token=None)).result == VerificationResult.INCONCLUSIVE
    assert verify_order_integrity(base_check(verifier_available=False)).result == VerificationResult.INCONCLUSIVE


def test_likelihood_directions_and_bayesian_update() -> None:
    requests = pd.DataFrame({
        "refund_request_id": ["a", "b", "c", "d", "e", "f"],
        "partition": ["train"] * 6,
        "is_refund_abuse_simulated": [True, True, True, False, False, False],
    })
    events = pd.DataFrame({
        "refund_request_id": ["a", "b", "c", "d", "e", "f"],
        "partition": ["train"] * 6,
        "result": ["inconsistent", "inconsistent", "inconclusive",
                   "consistent", "consistent", "inconclusive"],
    })
    table = estimate_likelihoods(requests, events, 1.0)
    assert table["likelihood_ratios"]["consistent"] < 1
    assert table["likelihood_ratios"]["inconsistent"] > 1
    assert 0.8 <= table["likelihood_ratios"]["inconclusive"] <= 1.25
    assert bayesian_update(0.5, table["likelihood_ratios"]["consistent"]) < 0.5
    assert bayesian_update(0.5, table["likelihood_ratios"]["inconsistent"]) > 0.5


def test_policy_is_deterministic() -> None:
    config = PolicyConfig(
        schema_version="1.0", policy_version="test", additional_loss_rate=0.35,
        salvage_rate=0.45, verification_cost_paise=2500, review_cost_paise=6000,
        legitimate_verification_friction_paise=4500,
        legitimate_review_friction_paise=12000, legitimate_delay_cost_paise=2500,
        reverse_logistics_cost_paise=9000, max_manual_review_rate=0.05,
        max_legitimate_delay_rate=0.2, laplace_alpha=1.0,
        inconclusive_lr_min=0.8, inconclusive_lr_max=1.25,
        output_dir="artifacts/test",
    )
    first = choose_action(0.4, 100000, config)
    second = choose_action(0.4, 100000, config)
    assert first == second
    assert first.action in set(RecommendedAction)


def test_policy_approves_probability_at_lower_threshold() -> None:
    config = PolicyConfig(
        schema_version="1.0", policy_version="test", additional_loss_rate=0.35,
        salvage_rate=0.45, verification_cost_paise=2500, review_cost_paise=6000,
        legitimate_verification_friction_paise=4500,
        legitimate_review_friction_paise=12000, legitimate_delay_cost_paise=2500,
        reverse_logistics_cost_paise=9000, max_manual_review_rate=0.05,
        max_legitimate_delay_rate=0.2, laplace_alpha=1.0,
        inconclusive_lr_min=0.8, inconclusive_lr_max=1.25,
        output_dir="artifacts/test",
    )
    contract = PolicyContract(
        policy_version="test", approve_threshold=0.1, review_threshold=0.9,
        review_amount_threshold_paise=0,
        max_manual_review_rate=0.05, max_legitimate_delay_rate=0.2,
    )
    actions = apply_contract(
        np.asarray([0.1], dtype=float), np.asarray([10_000], dtype=np.int64), contract, config
    )
    assert actions == [RecommendedAction.AUTO_APPROVE]

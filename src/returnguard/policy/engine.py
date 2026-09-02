from dataclasses import asdict, dataclass

from returnguard.config import PolicyConfig
from returnguard.domain.enums import RecommendedAction


@dataclass(frozen=True)
class PolicyDecision:
    action: RecommendedAction
    expected_cost_paise: float
    alternatives_paise: dict[str, float]
    rationale: str


def action_costs(
    probability: float, requested_amount_paise: int, config: PolicyConfig,
) -> dict[RecommendedAction, float]:
    exposure = requested_amount_paise * (1 + config.additional_loss_rate)
    approve = probability * exposure
    verify = (
        config.verification_cost_paise
        + (1 - probability) * config.legitimate_verification_friction_paise
        + probability * exposure * 0.28
    )
    return_first = (
        config.reverse_logistics_cost_paise
        + (1 - probability) * config.legitimate_delay_cost_paise
        + probability * exposure * (1 - config.salvage_rate)
    )
    review = (
        config.review_cost_paise
        + (1 - probability) * config.legitimate_review_friction_paise
    )
    return {
        RecommendedAction.AUTO_APPROVE: float(approve),
        RecommendedAction.VERIFY: float(verify),
        RecommendedAction.RETURN_FIRST: float(return_first),
        RecommendedAction.MANUAL_REVIEW: float(review),
    }


def choose_action(
    probability: float, requested_amount_paise: int, config: PolicyConfig,
) -> PolicyDecision:
    costs = action_costs(probability, requested_amount_paise, config)
    action = min(costs, key=lambda candidate: costs[candidate])
    return PolicyDecision(
        action=action, expected_cost_paise=costs[action],
        alternatives_paise={key.value: value for key, value in costs.items()},
        rationale=f"Minimum declared expected cost under {config.policy_version}.",
    )


def decision_dict(decision: PolicyDecision) -> dict[str, object]:
    return asdict(decision)

#!/usr/bin/env python3
"""
Check promotion gates for new models.
Evaluates whether new models meet the criteria for production deployment.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add algos to path
sys.path.append(str(Path(__file__).parent.parent))

from algos.core.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromotionGateChecker:
    """Check if new models meet promotion criteria."""

    def __init__(self, config_path: str):
        """Initialize gate checker."""
        self.config = load_config(config_path)
        self.gates = self.config.trading.learning.gates

        logger.info(f"Promotion gates: {dict(self.gates)}")

    def check_gates(self, evaluation_results: dict[str, Any], force: bool = False) -> dict[str, Any]:
        """
        Check if models pass promotion gates.
        
        Args:
            evaluation_results: Results from model evaluation
            force: Force promotion regardless of gates
            
        Returns:
            Dictionary with promotion decision and summary
        """
        if force:
            logger.info("🔓 Force promotion enabled - bypassing gates")
            return {
                "promote": True,
                "reason": "force_promotion",
                "summary": "Promotion forced by user override",
                "gates_passed": {},
                "metrics": {}
            }

        gates_passed = {}
        metrics = {}

        # Check classifier gates if available
        if "classifier" in evaluation_results:
            classifier_results = evaluation_results["classifier"]

            # Out-of-sample Sortino ratio
            oos_sortino = classifier_results.get("oos_sortino", 0.0)
            gates_passed["oos_sortino"] = oos_sortino >= self.gates.oos_sortino_min
            metrics["oos_sortino"] = oos_sortino

            # Out-of-sample profit factor
            oos_profit_factor = classifier_results.get("oos_profit_factor", 0.0)
            gates_passed["oos_profit_factor"] = oos_profit_factor >= self.gates.oos_profit_factor_min
            metrics["oos_profit_factor"] = oos_profit_factor

            # Maximum drawdown
            oos_max_dd = classifier_results.get("oos_max_dd", 1.0)
            gates_passed["oos_max_dd"] = oos_max_dd <= self.gates.oos_max_dd_max
            metrics["oos_max_dd"] = oos_max_dd

        # Check PPO gates if available
        if "ppo" in evaluation_results:
            ppo_results = evaluation_results["ppo"]

            # PPO reward improvement
            reward_improvement = ppo_results.get("reward_improvement", 0.0)
            gates_passed["ppo_reward"] = reward_improvement > 0.0
            metrics["ppo_reward_improvement"] = reward_improvement

            # PPO stability (low variance in rewards)
            reward_std = ppo_results.get("reward_std", float("inf"))
            gates_passed["ppo_stability"] = reward_std < 0.5  # Reasonable threshold
            metrics["ppo_reward_std"] = reward_std

        # Overall promotion decision
        all_gates_passed = all(gates_passed.values()) if gates_passed else False

        # Create summary
        summary = self._create_summary(gates_passed, metrics, all_gates_passed)

        decision = {
            "promote": all_gates_passed,
            "reason": "gates_passed" if all_gates_passed else "gates_failed",
            "summary": summary,
            "gates_passed": gates_passed,
            "metrics": metrics,
            "thresholds": {
                "oos_sortino_min": self.gates.oos_sortino_min,
                "oos_profit_factor_min": self.gates.oos_profit_factor_min,
                "oos_max_dd_max": self.gates.oos_max_dd_max,
            }
        }

        return decision

    def _create_summary(self, gates_passed: dict[str, bool], metrics: dict[str, float], overall: bool) -> str:
        """Create human-readable summary of gate results."""
        summary_lines = []

        if overall:
            summary_lines.append("✅ **PROMOTION APPROVED** - All gates passed")
        else:
            summary_lines.append("❌ **PROMOTION REJECTED** - One or more gates failed")

        summary_lines.append("")
        summary_lines.append("**Gate Results:**")

        for gate, passed in gates_passed.items():
            status = "✅" if passed else "❌"
            metric_value = metrics.get(gate, "N/A")

            if gate == "oos_sortino":
                threshold = self.gates.oos_sortino_min
                summary_lines.append(f"- {status} Sortino Ratio: {metric_value:.3f} (min: {threshold:.3f})")
            elif gate == "oos_profit_factor":
                threshold = self.gates.oos_profit_factor_min
                summary_lines.append(f"- {status} Profit Factor: {metric_value:.3f} (min: {threshold:.3f})")
            elif gate == "oos_max_dd":
                threshold = self.gates.oos_max_dd_max
                summary_lines.append(f"- {status} Max Drawdown: {metric_value:.3%} (max: {threshold:.3%})")
            elif gate == "ppo_reward":
                summary_lines.append(f"- {status} PPO Reward Improvement: {metric_value:.3f}")
            elif gate == "ppo_stability":
                summary_lines.append(f"- {status} PPO Stability (reward std): {metric_value:.3f}")
            else:
                summary_lines.append(f"- {status} {gate}: {metric_value}")

        if not gates_passed:
            summary_lines.append("")
            summary_lines.append("⚠️ No evaluation results available - cannot assess gates")

        return "\n".join(summary_lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check model promotion gates")
    parser.add_argument("--results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--force", action="store_true", help="Force promotion regardless of gates")
    parser.add_argument("--output", required=True, help="Path to output promotion decision JSON")

    args = parser.parse_args()

    # Load evaluation results
    try:
        with open(args.results) as f:
            evaluation_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load evaluation results: {e}")
        evaluation_results = {}

    # Check gates
    gate_checker = PromotionGateChecker(args.config)
    decision = gate_checker.check_gates(evaluation_results, args.force)

    # Save decision
    with open(args.output, "w") as f:
        json.dump(decision, f, indent=2)

    # Log results
    if decision["promote"]:
        logger.info("🚀 PROMOTION APPROVED")
    else:
        logger.info("❌ PROMOTION REJECTED")

    print(decision["summary"])

    return 0 if decision["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())

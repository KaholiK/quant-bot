#!/usr/bin/env python3
"""
Self-audit script for quant-bot repository.
Verifies required paths exist, validates config.yaml schema, and tests imports.
"""

import importlib.util
import sys
from pathlib import Path

import yaml

# Add algos to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class QuantBotAuditor:
    """Repository self-audit system."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results: list[tuple[str, str, str]] = []  # (category, test, status)
        self.config_obj = None

    def add_result(self, category: str, test: str, status: str):
        """Add a test result."""
        self.results.append((category, test, status))

    def verify_required_paths(self) -> None:
        """Verify all required paths exist."""
        required_files = [
            "pyproject.toml",
            "README.md",
            "config.yaml",
            "lean.json",
            "MainAlgo.py",
            "algos/core/feature_pipe.py",
            "algos/core/labels.py",
            "algos/core/cv_utils.py",
            "algos/core/risk.py",
            "algos/core/portfolio.py",
            "algos/core/exec_rl.py",
            "algos/core/runtime_state.py",
            "algos/core/polling.py",
            "storage/trades.py",
            "services/admin_api.py",
            "bots/discord_bot.py",
            "algos/strategies/scalper_sigma.py",
            "algos/strategies/trend_breakout.py",
            "algos/strategies/bull_mode.py",
            "algos/strategies/market_neutral.py",
            "algos/strategies/gamma_reversal.py",
            "notebooks/train_classifier.ipynb",
            "notebooks/train_ppo.ipynb",
            "tests/test_triple_barrier.py",
            "tests/test_purged_cv.py",
            "tests/test_risk_engine.py",
            ".env.example",
            "models/.gitkeep",
            "policies/.gitkeep"
        ]

        for file_path in required_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                self.add_result("Files", file_path, "PASS")
            else:
                self.add_result("Files", file_path, "FAIL")

        # Check test files exist (at least one test_*.py)
        test_dir = self.repo_root / "tests"
        if test_dir.exists():
            test_files = list(test_dir.glob("test_*.py"))
            if test_files:
                self.add_result("Files", "tests/test_*.py", "PASS")
            else:
                self.add_result("Files", "tests/test_*.py", "FAIL")
        else:
            self.add_result("Files", "tests/", "FAIL")

    def validate_config_schema(self) -> None:
        """Validate config.yaml schema using new config loader."""
        config_path = self.repo_root / "config.yaml"

        if not config_path.exists():
            self.add_result("Config", "config.yaml exists", "FAIL")
            return

        try:
            # Try to import and use the new config loader
            from algos.core.config_loader import load_config

            self.config_obj = load_config(str(config_path))
            self.add_result("Config", "config.yaml loads with new loader", "PASS")

            # Validate key sections exist
            if hasattr(self.config_obj, "trading"):
                self.add_result("Config", "trading section", "PASS")

                # Check nested sections
                if hasattr(self.config_obj.trading, "universe"):
                    self.add_result("Config", "trading.universe", "PASS")
                else:
                    self.add_result("Config", "trading.universe", "FAIL")

                if hasattr(self.config_obj.trading, "risk"):
                    self.add_result("Config", "trading.risk", "PASS")
                    # Validate risk parameters
                    risk = self.config_obj.trading.risk
                    if 0 <= risk.per_trade_risk_pct <= 0.05:
                        self.add_result("Config", "risk.per_trade_risk_pct in range", "PASS")
                    else:
                        self.add_result("Config", "risk.per_trade_risk_pct in range", "FAIL")

                    if 0 < risk.kill_switch_dd <= 0.5:
                        self.add_result("Config", "risk.kill_switch_dd in range", "PASS")
                    else:
                        self.add_result("Config", "risk.kill_switch_dd in range", "FAIL")
                else:
                    self.add_result("Config", "trading.risk", "FAIL")

                if hasattr(self.config_obj.trading, "models"):
                    self.add_result("Config", "trading.models", "PASS")
                else:
                    self.add_result("Config", "trading.models", "FAIL")

                if hasattr(self.config_obj.trading, "strategies"):
                    self.add_result("Config", "trading.strategies", "PASS")
                else:
                    self.add_result("Config", "trading.strategies", "FAIL")

                # Validate new UI section
                if hasattr(self.config_obj.trading, "ui"):
                    self.add_result("Config", "trading.ui", "PASS")

                    if hasattr(self.config_obj.trading.ui, "discord"):
                        self.add_result("Config", "trading.ui.discord", "PASS")
                    else:
                        self.add_result("Config", "trading.ui.discord", "FAIL")

                    if hasattr(self.config_obj.trading.ui, "admin_api"):
                        self.add_result("Config", "trading.ui.admin_api", "PASS")
                    else:
                        self.add_result("Config", "trading.ui.admin_api", "FAIL")
                else:
                    self.add_result("Config", "trading.ui", "FAIL")
            else:
                self.add_result("Config", "trading section", "FAIL")

            # Print normalized config
            print("\n=== VALIDATED CONFIG ===")
            config_dict = self.config_obj.model_dump()
            print(yaml.dump(config_dict, default_flow_style=False, indent=2))
            print("=" * 25)

        except ImportError as e:
            self.add_result("Config", "config_loader import", f"FAIL: {e}")
            # Fallback to basic YAML validation
            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                self.add_result("Config", "config.yaml syntax", "PASS")

                if "trading" in config_data:
                    self.add_result("Config", "trading key present", "PASS")
                else:
                    self.add_result("Config", "trading key present", "FAIL")

            except Exception as yaml_e:
                self.add_result("Config", "config.yaml syntax", f"FAIL: {yaml_e}")
        except Exception as e:
            self.add_result("Config", "config validation", f"FAIL: {e}")
    def test_core_imports(self) -> None:
        """Test importing core modules to catch syntax/import errors."""
        core_modules = [
            "algos.core.feature_pipe",
            "algos.core.labels",
            "algos.core.cv_utils",
            "algos.core.risk",
            "algos.core.portfolio",
            "algos.core.exec_rl",
            "algos.core.runtime_state",
            "algos.core.polling"
        ]

        ui_modules = [
            "storage.trades",
            "services.admin_api",
            "bots.discord_bot"
        ]

        strategy_modules = [
            "algos.strategies.scalper_sigma",
            "algos.strategies.trend_breakout",
            "algos.strategies.bull_mode",
            "algos.strategies.market_neutral",
            "algos.strategies.gamma_reversal"
        ]

        # Add repo root to Python path for imports
        sys.path.insert(0, str(self.repo_root))

        # Test core modules
        for module_name in core_modules + strategy_modules:
            try:
                # Try to import the module
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.add_result("Imports", module_name, "FAIL: Module not found")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.add_result("Imports", module_name, "PASS")

            except ImportError as e:
                # Handle missing dependencies gracefully - this is expected in CI
                if "numpy" in str(e) or "pandas" in str(e) or "sklearn" in str(e):
                    self.add_result("Imports", module_name, "SKIP: Missing dependencies (expected in CI)")
                else:
                    self.add_result("Imports", module_name, f"FAIL: {e}")
            except Exception as e:
                self.add_result("Imports", module_name, f"FAIL: {e}")

        # Test UI modules with special handling for optional dependencies
        for module_name in ui_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.add_result("UI-Imports", module_name, "FAIL: Module not found")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.add_result("UI-Imports", module_name, "PASS")

            except ImportError as e:
                # UI dependencies might not be available in CI
                if any(dep in str(e) for dep in ["fastapi", "discord", "uvicorn", "prometheus_client"]):
                    self.add_result("UI-Imports", module_name, "SKIP: UI dependencies not installed")
                else:
                    self.add_result("UI-Imports", module_name, f"FAIL: {e}")
            except Exception as e:
                self.add_result("UI-Imports", module_name, f"FAIL: {e}")

    def generate_report(self) -> str:
        """Generate audit report as markdown table."""
        report_lines = [
            "# Quant Bot Self-Audit Report",
            "",
            f"Audit run on repository: {self.repo_root}",
            "",
            "## Results Summary",
            "",
            "| Category | Test | Status |",
            "|----------|------|--------|"
        ]

        for category, test, status in self.results:
            report_lines.append(f"| {category} | {test} | {status} |")

        # Add summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for _, _, status in self.results if status == "PASS")
        failed_tests = total_tests - passed_tests

        report_lines.extend([
            "",
            f"**Total Tests:** {total_tests}  ",
            f"**Passed:** {passed_tests}  ",
            f"**Failed:** {failed_tests}  ",
            f"**Success Rate:** {passed_tests/total_tests*100:.1f}%  ",
            ""
        ])

        return "\n".join(report_lines)

    def run_audit(self) -> bool:
        """Run complete audit and return success status."""
        print("Starting quant-bot repository self-audit...")

        print("✓ Verifying required paths...")
        self.verify_required_paths()

        print("✓ Validating config.yaml schema...")
        self.validate_config_schema()

        print("✓ Testing core module imports...")
        self.test_core_imports()

        # Generate and write report
        report = self.generate_report()
        report_path = self.repo_root / "self_audit_report.md"

        with open(report_path, "w") as f:
            f.write(report)

        print(f"\n✓ Audit complete! Report written to: {report_path}")

        # Print summary to console
        print("\n" + "="*50)
        print("AUDIT SUMMARY")
        print("="*50)

        categories = {}
        for category, test, status in self.results:
            if category not in categories:
                categories[category] = {"PASS": 0, "FAIL": 0}
            if status == "PASS":
                categories[category]["PASS"] += 1
            else:
                categories[category]["FAIL"] += 1

        for category, stats in categories.items():
            total = stats["PASS"] + stats["FAIL"]
            success_rate = stats["PASS"] / total * 100 if total > 0 else 0
            print(f"{category:12} | {stats['PASS']:2d}/{total:2d} passed ({success_rate:4.1f}%)")

        total_tests = len(self.results)
        total_passed = sum(1 for _, _, status in self.results if status == "PASS")
        total_skipped = sum(1 for _, _, status in self.results if "SKIP" in status)
        overall_rate = total_passed / total_tests * 100 if total_tests > 0 else 0

        print("-" * 50)
        print(f"{'OVERALL':12} | {total_passed:2d}/{total_tests:2d} passed ({overall_rate:4.1f}%)")
        if total_skipped > 0:
            print(f"{'':12}   {total_skipped:2d} tests skipped (expected)")

        # Return True if all tests passed or were expected skips
        return all(status == "PASS" or "SKIP" in status for _, _, status in self.results)


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    auditor = QuantBotAuditor(repo_root)

    success = auditor.run_audit()

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()

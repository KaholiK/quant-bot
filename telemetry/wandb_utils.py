"""
Weights & Biases (W&B) telemetry integration.
No-op if WANDB_API_KEY is not configured.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings


class WandBUtils:
    """W&B integration with graceful fallback."""

    def __init__(self):
        """Initialize W&B if API key is available."""
        self.enabled = settings.has_wandb()
        self.run = None

        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                logger.info("W&B integration enabled")
            except ImportError:
                logger.warning("wandb package not installed, disabling W&B")
                self.enabled = False
        else:
            logger.info("W&B API key not configured, telemetry disabled")

    def init_run(
        self,
        project: str = "quantbot",
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None
    ) -> bool:
        """
        Initialize W&B run.
        
        Args:
            project: W&B project name
            name: Run name (optional)
            tags: List of tags
            config: Configuration dictionary
            
        Returns:
            True if initialized successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            self.run = self.wandb.init(
                project=project,
                name=name,
                tags=tags or ["paper"],
                config=config or {},
                mode="online" if settings.APP_ENV == "prod" else "offline"
            )
            logger.info(f"W&B run initialized: {self.run.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            return False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.enabled or not self.run:
            return

        try:
            self.wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")

    def log_artifact(
        self,
        path: str,
        name: str | None = None,
        artifact_type: str = "dataset"
    ) -> None:
        """
        Log artifact (file/directory) to W&B.
        
        Args:
            path: Path to file or directory
            name: Artifact name (defaults to filename)
            artifact_type: Type of artifact
        """
        if not self.enabled or not self.run:
            return

        try:
            artifact_name = name or Path(path).name
            artifact = self.wandb.Artifact(artifact_name, type=artifact_type)

            if Path(path).is_dir():
                artifact.add_dir(path)
            else:
                artifact.add_file(path)

            self.run.log_artifact(artifact)
            logger.info(f"Logged artifact to W&B: {artifact_name}")
        except Exception as e:
            logger.error(f"Failed to log artifact to W&B: {e}")

    def finish(self) -> None:
        """Finish W&B run."""
        if not self.enabled or not self.run:
            return

        try:
            self.run.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log summary metrics at end of run.
        
        Args:
            summary: Dictionary of summary metrics
        """
        if not self.enabled or not self.run:
            return

        try:
            for key, value in summary.items():
                self.run.summary[key] = value
            logger.info("Logged summary to W&B")
        except Exception as e:
            logger.error(f"Failed to log summary to W&B: {e}")


# Global instance
_wandb_utils = None


def get_wandb() -> WandBUtils:
    """Get global W&B utils instance."""
    global _wandb_utils
    if _wandb_utils is None:
        _wandb_utils = WandBUtils()
    return _wandb_utils

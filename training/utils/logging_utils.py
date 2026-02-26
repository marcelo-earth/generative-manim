"""W&B setup and logging utilities."""

import os
import wandb
from omegaconf import DictConfig, OmegaConf


def setup_wandb(
    config: DictConfig,
    run_name: str | None = None,
    tags: list[str] | None = None,
) -> wandb.sdk.wandb_run.Run | None:
    """Initialize W&B run from config."""
    wandb_config = config.get("wandb", {})
    project = wandb_config.get("project", "generative-manim")
    entity = wandb_config.get("entity", None)
    default_tags = list(wandb_config.get("tags", []))

    all_tags = default_tags + (tags or [])

    # Convert OmegaConf to plain dict for W&B
    flat_config = OmegaConf.to_container(config, resolve=True)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=all_tags,
        config=flat_config,
    )
    return run


def log_metrics(metrics: dict, step: int | None = None):
    """Log metrics to W&B if active."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_table(name: str, columns: list[str], data: list[list]):
    """Log a W&B table."""
    if wandb.run is not None:
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})


def finish_wandb():
    """Finish W&B run."""
    if wandb.run is not None:
        wandb.finish()

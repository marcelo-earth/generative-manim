"""Config loading with OmegaConf: merges base + model + stage configs."""

import os
from pathlib import Path

from omegaconf import OmegaConf, DictConfig


CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_config(
    model_name: str | None = None,
    stage: str | None = None,
    overrides: dict | None = None,
) -> DictConfig:
    """
    Load and merge configuration files.

    Merge order (later overrides earlier):
    1. config/base.yaml (always loaded)
    2. config/models/{model_name}.yaml (if model specified)
    3. config/{stage}.yaml (if stage specified)
    4. CLI overrides dict
    """
    configs = []

    # Base config
    base_path = CONFIG_DIR / "base.yaml"
    if base_path.exists():
        configs.append(OmegaConf.load(base_path))

    # Model config
    if model_name:
        model_path = CONFIG_DIR / "models" / f"{model_name}.yaml"
        if model_path.exists():
            configs.append(OmegaConf.load(model_path))
        else:
            raise FileNotFoundError(f"Model config not found: {model_path}")

    # Stage config
    if stage:
        stage_path = CONFIG_DIR / f"{stage}.yaml"
        if stage_path.exists():
            configs.append(OmegaConf.load(stage_path))
        else:
            raise FileNotFoundError(f"Stage config not found: {stage_path}")

    # CLI overrides
    if overrides:
        configs.append(OmegaConf.create(overrides))

    # Merge all
    merged = OmegaConf.merge(*configs) if configs else OmegaConf.create({})
    return merged


def get_model_names() -> list[str]:
    """List available model config names."""
    models_dir = CONFIG_DIR / "models"
    if not models_dir.exists():
        return []
    return [p.stem for p in models_dir.glob("*.yaml")]

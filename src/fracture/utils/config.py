"""Lightweight, composable YAML config loading built on OmegaConf.

We deliberately avoid full Hydra: no working-directory rewriting, no global
state.  ``load_config`` merges, in increasing priority:

1. ``configs/config.yaml`` (the base)
2. the group files named in the base ``defaults:`` list.  Each entry is either
   a mapping ``{group: name}`` or a string ``"group: name"`` and resolves to
   ``configs/<group>/<name>.yaml`` merged under key ``group``.
3. an optional experiment file (``configs/experiment/<name>.yaml``)
4. dotlist overrides from the command line (``model.lr=1e-4``).  An override of
   the form ``group=name`` (where ``configs/group/`` exists) instead swaps the
   group file, e.g. ``model=convnext_tiny``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def _normalise_defaults(raw) -> dict[str, str]:
    """Turn a ``defaults`` list into an ordered ``{group: name}`` mapping."""
    out: dict[str, str] = {}
    for entry in raw or []:
        if isinstance(entry, str):
            group, name = (p.strip() for p in entry.split(":", 1))
        elif isinstance(entry, dict) or OmegaConf.is_dict(entry):
            (group, name), *_ = list(dict(entry).items())
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported defaults entry: {entry!r}")
        out[str(group)] = str(name)
    return out


def _load_group_defaults(defaults: dict[str, str], config_root: Path) -> DictConfig:
    merged: DictConfig = OmegaConf.create({})  # type: ignore[assignment]
    for group, name in defaults.items():
        path = config_root / group / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"config default not found: {path}")
        merged = OmegaConf.merge(merged, {group: OmegaConf.load(path)})  # type: ignore[assignment]
    return merged


def _split_group_overrides(overrides: Sequence[str], config_root: Path) -> tuple[dict[str, str], list[str]]:
    groups: dict[str, str] = {}
    dotlist: list[str] = []
    for ov in overrides:
        if "=" in ov:
            key, val = ov.split("=", 1)
            if "." not in key and (config_root / key).is_dir():
                groups[key] = val
                continue
        dotlist.append(ov)
    return groups, dotlist


def load_config(
    overrides: Sequence[str] | None = None,
    *,
    experiment: str | None = None,
    config_root: Path | None = None,
) -> DictConfig:
    config_root = config_root or CONFIG_ROOT
    base: DictConfig = OmegaConf.load(config_root / "config.yaml")  # type: ignore[assignment]

    group_sel, dotlist = _split_group_overrides(list(overrides or []), config_root)
    defaults = _normalise_defaults(base.pop("defaults", []))
    defaults.update(group_sel)  # CLI group selectors win

    cfg: DictConfig = OmegaConf.merge(_load_group_defaults(defaults, config_root), base)  # type: ignore[assignment]

    experiment = experiment or cfg.get("experiment")
    if experiment:
        exp_path = config_root / "experiment" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"experiment config not found: {exp_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_path))  # type: ignore[assignment]

    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))  # type: ignore[assignment]

    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)

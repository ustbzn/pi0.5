#!/usr/bin/env python3
"""
VS Code-friendly debug launcher for DrivePi workflows.

Preferred setup:
  1. Fill the path overrides in `.vscode/debug.env` when your local paths differ.
  2. Pick one of the Run & Debug presets from `.vscode/launch.json`.

Examples:
  python script/vscode_debug.py --mode train --print-config --no-run
  python script/vscode_debug.py --mode train --n-updates 2 --per-device-batch-size 1
  python script/vscode_debug.py --mode infer --max-samples 1
  python script/vscode_debug.py --mode eval
  python script/vscode_debug.py --mode train --set resume_checkpoint_path=/path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs_debug"
DEFAULT_TRAIN_CONFIG = (
    REPO_ROOT
    / "config/train/DrivePi0/0421_qwen3_naviv2_9dim_smoothLoss_lora_vlm_1gpu_local.yaml"
)
DEFAULT_INFER_CONFIG = REPO_ROOT / "config/inference/DrivePi0/qwen_mogo_havetrain_vlm.yaml"
DEFAULT_EVAL_CONFIG = DEFAULT_INFER_CONFIG

MODE_CONFIG_ENV = {
    "train": "DRIVEPI_DEBUG_TRAIN_CONFIG",
    "infer": "DRIVEPI_DEBUG_INFER_CONFIG",
    "eval": "DRIVEPI_DEBUG_EVAL_CONFIG",
}

DEFAULT_CONFIG_PATH = {
    "train": DEFAULT_TRAIN_CONFIG,
    "infer": DEFAULT_INFER_CONFIG,
    "eval": DEFAULT_EVAL_CONFIG,
}

_NOW = datetime.now()


def _env_str(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_path(name: str) -> Path | None:
    value = _env_str(name)
    return Path(value).expanduser() if value else None


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _path_or_none(path: Path | None) -> Path | None:
    return _resolve_path(path) if path is not None else None


def _arg_or_env_path(arg_path: Path | None, env_name: str) -> Path | None:
    if arg_path is not None:
        return _resolve_path(arg_path)
    env_path = _env_path(env_name)
    return _path_or_none(env_path)


def _default_log_dir() -> Path:
    return _arg_or_env_path(None, "LOG_DIR") or DEFAULT_LOG_DIR


def _register_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    OmegaConf.register_new_resolver("now", lambda fmt: _NOW.strftime(fmt), replace=True)


def _ensure_env(log_dir: Path) -> None:
    resolved_log_dir = _resolve_path(log_dir)
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{REPO_ROOT}:{pythonpath}" if pythonpath else str(REPO_ROOT)
    )
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ["LOG_DIR"] = str(resolved_log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)


def _default_config_path(mode: str) -> Path:
    env_name = MODE_CONFIG_ENV[mode]
    env_value = _arg_or_env_path(None, env_name)
    return env_value or DEFAULT_CONFIG_PATH[mode]


def _load_config(path: Path) -> OmegaConf:
    resolved_path = _resolve_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {resolved_path}")
    return OmegaConf.load(resolved_path)


def _update_if_present(cfg: OmegaConf, key: str, value) -> None:
    if OmegaConf.select(cfg, key) is not None:
        OmegaConf.update(cfg, key, value, merge=True)


def _apply_common_debug_overrides(cfg: OmegaConf, args: argparse.Namespace) -> None:
    with open_dict(cfg):
        cfg.current_dir = str(REPO_ROOT)
        cfg.gpu_id = args.gpu
        cfg.multi_gpu = False
        if "n_nodes" in cfg:
            cfg.n_nodes = 1
        if "num_workers" in cfg:
            cfg.num_workers = 0
        if "debug" in cfg:
            cfg.debug = True
        if "use_torch_compile" in cfg and not args.enable_compile:
            cfg.use_torch_compile = False
        if "use_amp" in cfg and not args.enable_amp:
            cfg.use_amp = False

    _update_if_present(cfg, "data.work_dir", str(REPO_ROOT))
    _update_if_present(cfg, "data.train.work_dir", str(REPO_ROOT))
    _update_if_present(cfg, "data.val.work_dir", str(REPO_ROOT))

    if cfg.get("swanlab") is not None and not args.enable_swanlab:
        OmegaConf.update(cfg, "swanlab.enable", False, merge=True)

    pretrained_model_path = _arg_or_env_path(
        args.pretrained_model_path, "DRIVEPI_DEBUG_PRETRAINED_MODEL_PATH"
    )
    if pretrained_model_path:
        model_path = str(pretrained_model_path)
        _update_if_present(cfg, "pretrained_model_path", model_path)
        _update_if_present(cfg, "joint.config.pretrained_model_path", model_path)


def _apply_train_debug_overrides(
    cfg: OmegaConf, args: argparse.Namespace
) -> OmegaConf:
    with open_dict(cfg):
        cfg.per_device_batch_size = args.per_device_batch_size
        cfg.global_batch_size = args.per_device_batch_size
        cfg.n_updates = args.n_updates
        cfg.save_model_freq = max(args.n_updates, 1)
        cfg.log_freq = 1
        if "eval_freq" in cfg:
            cfg.eval_freq = args.n_updates + 1
        if "eval_size" in cfg:
            cfg.eval_size = args.per_device_batch_size

    train_data_dir = _arg_or_env_path(
        args.train_data_dir, "DRIVEPI_DEBUG_TRAIN_DATA_DIR"
    )
    statistics_path = _arg_or_env_path(
        args.statistics_path, "DRIVEPI_DEBUG_STATISTICS_PATH"
    )

    if train_data_dir:
        data_root = str(train_data_dir)
        _update_if_present(cfg, "data.train.dynamic_dir", data_root)
        _update_if_present(cfg, "data.val.dynamic_dir", data_root)
        if not statistics_path:
            auto_statistics = train_data_dir / "statistics.json"
            _update_if_present(
                cfg, "data.train.statistics_path", str(auto_statistics)
            )
            _update_if_present(cfg, "data.val.statistics_path", str(auto_statistics))

    if statistics_path:
        statistics_path_str = str(statistics_path)
        _update_if_present(cfg, "data.train.statistics_path", statistics_path_str)
        _update_if_present(cfg, "data.val.statistics_path", statistics_path_str)

    return cfg


def _make_train_scheduler_debug_safe(cfg: OmegaConf) -> None:
    for scheduler_key in ("action_lr_scheduler", "vlm_lr_scheduler"):
        first_cycle_steps = OmegaConf.select(cfg, f"{scheduler_key}.first_cycle_steps")
        warmup_steps = OmegaConf.select(cfg, f"{scheduler_key}.warmup_steps")
        if first_cycle_steps is None or warmup_steps is None:
            continue

        first_cycle_steps = max(int(first_cycle_steps), 1)
        warmup_steps = int(warmup_steps)
        safe_warmup_steps = min(max(warmup_steps, 0), first_cycle_steps - 1)

        if first_cycle_steps != OmegaConf.select(
            cfg, f"{scheduler_key}.first_cycle_steps"
        ):
            OmegaConf.update(
                cfg, f"{scheduler_key}.first_cycle_steps", first_cycle_steps, merge=True
            )

        if safe_warmup_steps != warmup_steps:
            print(
                "[vscode_debug] "
                f"{scheduler_key}.warmup_steps={warmup_steps} is too large for "
                f"first_cycle_steps={first_cycle_steps}; using {safe_warmup_steps} "
                "for debug."
            )
            OmegaConf.update(
                cfg, f"{scheduler_key}.warmup_steps", safe_warmup_steps, merge=True
            )


def _apply_infer_debug_overrides(
    cfg: OmegaConf, args: argparse.Namespace
) -> OmegaConf:
    checkpoint_path = _arg_or_env_path(
        args.checkpoint_path, "DRIVEPI_DEBUG_CHECKPOINT_PATH"
    )
    statistics_path = _arg_or_env_path(
        args.statistics_path, "DRIVEPI_DEBUG_STATISTICS_PATH"
    )

    if checkpoint_path:
        _update_if_present(cfg, "checkpoint_path", str(checkpoint_path))

    if statistics_path:
        _update_if_present(cfg, "data.statistics_path", str(statistics_path))

    return cfg


def _apply_eval_debug_overrides(
    cfg: OmegaConf, args: argparse.Namespace
) -> OmegaConf:
    checkpoint_path = _arg_or_env_path(
        args.checkpoint_path, "DRIVEPI_DEBUG_CHECKPOINT_PATH"
    )
    statistics_path = _arg_or_env_path(
        args.statistics_path, "DRIVEPI_DEBUG_STATISTICS_PATH"
    )

    with open_dict(cfg):
        if "device_batch_size" in cfg:
            cfg.device_batch_size = 1
        if "save_trajectories" in cfg:
            cfg.save_trajectories = False
        if "num_workers" in cfg:
            cfg.num_workers = 0

    if checkpoint_path:
        _update_if_present(cfg, "checkpoint_path", str(checkpoint_path))

    if statistics_path:
        _update_if_present(cfg, "data.statistics_path", str(statistics_path))

    return cfg


def _apply_user_overrides(cfg: OmegaConf, overrides: list[str]) -> OmegaConf:
    if not overrides:
        return cfg
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def _resolve_config(cfg: OmegaConf) -> OmegaConf:
    OmegaConf.resolve(cfg)
    return cfg


def _print_summary(mode: str, config_path: Path, cfg: OmegaConf) -> None:
    print(f"[vscode_debug] mode={mode}")
    print(f"[vscode_debug] config={_resolve_path(config_path)}")
    if cfg.get("log_dir") is not None:
        print(f"[vscode_debug] log_dir={cfg.log_dir}")
    if cfg.get("pretrained_model_path") is not None:
        print(f"[vscode_debug] pretrained_model_path={cfg.pretrained_model_path}")
    if cfg.get("checkpoint_path") is not None:
        print(f"[vscode_debug] checkpoint_path={cfg.checkpoint_path}")
    for key in (
        "data.statistics_path",
        "data.train.statistics_path",
        "data.val.statistics_path",
        "data.train.dynamic_dir",
        "data.val.dynamic_dir",
    ):
        value = OmegaConf.select(cfg, key)
        if value is not None:
            print(f"[vscode_debug] {key}={value}")


def _get_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _raise_actionable_import_error(exc: ImportError) -> None:
    message = str(exc)
    if "Qwen3VLProcessor" in message:
        raise RuntimeError(
            "Current Python environment does not provide "
            "`transformers.Qwen3VLProcessor`. "
            "Use the project's Python 3.10 environment and install the repo "
            "dependencies before running this debug script."
        ) from exc
    raise RuntimeError(
        "Current Python environment is missing project dependencies. "
        f"Original import error: {message}"
    ) from exc


def _validate_existing_path(
    path_value: str | None, *, label: str, config_key: str, env_var: str | None = None
) -> None:
    if not path_value:
        return
    candidate = _resolve_path(path_value)
    if candidate.exists():
        return

    guidance = [f"{label} does not exist: {candidate}"]
    if env_var:
        guidance.append(f"set `{env_var}` in `.vscode/debug.env`")
    guidance.append(f"or update `{config_key}` in the selected config")
    raise RuntimeError("; ".join(guidance) + ".")


def _validate_train_paths(cfg: OmegaConf) -> None:
    _validate_existing_path(
        OmegaConf.select(cfg, "pretrained_model_path"),
        label="Pretrained model path",
        config_key="pretrained_model_path",
        env_var="DRIVEPI_DEBUG_PRETRAINED_MODEL_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "joint.config.pretrained_model_path"),
        label="Joint model pretrained path",
        config_key="joint.config.pretrained_model_path",
        env_var="DRIVEPI_DEBUG_PRETRAINED_MODEL_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "data.train.dynamic_dir"),
        label="Train dataset root",
        config_key="data.train.dynamic_dir",
        env_var="DRIVEPI_DEBUG_TRAIN_DATA_DIR",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "data.train.statistics_path"),
        label="Train statistics path",
        config_key="data.train.statistics_path",
        env_var="DRIVEPI_DEBUG_STATISTICS_PATH",
    )


def _validate_infer_paths(cfg: OmegaConf, args: argparse.Namespace) -> Path:
    _validate_existing_path(
        OmegaConf.select(cfg, "pretrained_model_path"),
        label="Pretrained model path",
        config_key="pretrained_model_path",
        env_var="DRIVEPI_DEBUG_PRETRAINED_MODEL_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "joint.config.pretrained_model_path"),
        label="Joint model pretrained path",
        config_key="joint.config.pretrained_model_path",
        env_var="DRIVEPI_DEBUG_PRETRAINED_MODEL_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "checkpoint_path"),
        label="Checkpoint path",
        config_key="checkpoint_path",
        env_var="DRIVEPI_DEBUG_CHECKPOINT_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "data.statistics_path"),
        label="Inference statistics path",
        config_key="data.statistics_path",
        env_var="DRIVEPI_DEBUG_STATISTICS_PATH",
    )

    val_data_dir = _arg_or_env_path(args.val_data_dir, "DRIVEPI_DEBUG_VAL_DATA_DIR")
    if val_data_dir is None:
        raise RuntimeError(
            "Inference requires a validation directory; set "
            "`DRIVEPI_DEBUG_VAL_DATA_DIR` in `.vscode/debug.env` or pass "
            "`--val-data-dir`."
        )
    _validate_existing_path(
        str(val_data_dir),
        label="Inference validation directory",
        config_key="--val-data-dir",
        env_var="DRIVEPI_DEBUG_VAL_DATA_DIR",
    )
    return val_data_dir


def _validate_eval_paths(cfg: OmegaConf) -> None:
    _validate_existing_path(
        OmegaConf.select(cfg, "pretrained_model_path"),
        label="Pretrained model path",
        config_key="pretrained_model_path",
        env_var="DRIVEPI_DEBUG_PRETRAINED_MODEL_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "joint.config.pretrained_model_path"),
        label="Joint model pretrained path",
        config_key="joint.config.pretrained_model_path",
        env_var="DRIVEPI_DEBUG_PRETRAINED_MODEL_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "checkpoint_path"),
        label="Checkpoint path",
        config_key="checkpoint_path",
        env_var="DRIVEPI_DEBUG_CHECKPOINT_PATH",
    )
    _validate_existing_path(
        OmegaConf.select(cfg, "data.statistics_path"),
        label="Eval statistics path",
        config_key="data.statistics_path",
        env_var="DRIVEPI_DEBUG_STATISTICS_PATH",
    )


def _run_train(cfg: OmegaConf) -> None:
    _validate_train_paths(cfg)
    try:
        agent_cls = _get_class(cfg._target_)
    except ImportError as exc:
        _raise_actionable_import_error(exc)
    agent = agent_cls(cfg)
    agent.run()


def _run_infer(cfg: OmegaConf, args: argparse.Namespace) -> None:
    val_data_dir = _validate_infer_paths(cfg, args)
    output_dir = _arg_or_env_path(args.output_dir, "DRIVEPI_DEBUG_OUTPUT_DIR")
    if output_dir is None:
        output_dir = _resolve_path(Path(os.environ["LOG_DIR"]) / "inference")

    try:
        from src.agent.drivepi0.inference_mogo_test import DrivePiZeroInferenceAgent
    except ImportError as exc:
        _raise_actionable_import_error(exc)

    statistics_path = _resolve_path(Path(cfg.data.statistics_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vscode_debug] val_data_dir={val_data_dir}")
    print(f"[vscode_debug] statistics_path={statistics_path}")
    print(f"[vscode_debug] output_dir={output_dir}")
    print(f"[vscode_debug] max_samples={args.max_samples}")

    inference_agent = DrivePiZeroInferenceAgent(cfg)
    inference_agent.run_batch_inference(
        val_data_dir=str(val_data_dir),
        output_dir=str(output_dir),
        statistic_path=str(statistics_path),
        max_samples=args.max_samples,
        rank=0,
        world_size=1,
    )


def _run_eval(cfg: OmegaConf) -> None:
    _validate_eval_paths(cfg)
    try:
        agent_cls = _get_class(cfg._target_)
    except ImportError as exc:
        _raise_actionable_import_error(exc)
    agent = agent_cls(cfg)
    agent.run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-process debug launcher for DrivePi in VS Code."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "eval"],
        default="train",
        help="train/infer/eval debug mode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional custom config path. Defaults to the mode-specific VS Code env or repo default.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id to use.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_default_log_dir(),
        help="Base log directory used to populate LOG_DIR.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=Path,
        default=None,
        help="Override pretrained_model_path and joint.config.pretrained_model_path.",
    )
    parser.add_argument(
        "--statistics-path",
        type=Path,
        default=None,
        help="Override statistics.json path.",
    )
    parser.add_argument(
        "--train-data-dir",
        type=Path,
        default=None,
        help="Override dataset root for training. Expected to contain train/ and val/.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Override checkpoint_path for inference/eval config.",
    )
    parser.add_argument(
        "--val-data-dir",
        type=Path,
        default=None,
        help="Batch inference input directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Batch inference output directory. Defaults to LOG_DIR/inference.",
    )
    parser.add_argument(
        "--n-updates",
        type=int,
        default=1,
        help="Train mode only. Default keeps debugging short and deterministic.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Train mode only. global_batch_size will match this value in debug mode.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Infer mode only. Limits processed samples for quick debugging.",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Keep torch.compile enabled instead of forcing it off.",
    )
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        help="Keep AMP enabled instead of forcing it off.",
    )
    parser.add_argument(
        "--enable-swanlab",
        action="store_true",
        help="Keep SwanLab enabled instead of forcing it off.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Extra OmegaConf dotlist override, e.g. --set resume_checkpoint_path=/path/to/ckpt.pt",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully resolved config before running.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Resolve and print config, but do not instantiate the model.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _register_resolvers()
    _ensure_env(args.log_dir)

    config_path = args.config or _default_config_path(args.mode)
    cfg = _load_config(config_path)
    _apply_common_debug_overrides(cfg, args)

    if args.mode == "train":
        _apply_train_debug_overrides(cfg, args)
    elif args.mode == "infer":
        _apply_infer_debug_overrides(cfg, args)
    elif args.mode == "eval":
        _apply_eval_debug_overrides(cfg, args)

    cfg = _apply_user_overrides(cfg, args.overrides)
    cfg = _resolve_config(cfg)
    if args.mode == "train":
        _make_train_scheduler_debug_safe(cfg)

    _print_summary(args.mode, config_path, cfg)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    if args.no_run:
        return

    if args.mode == "train":
        _run_train(cfg)
        return
    if args.mode == "infer":
        _run_infer(cfg, args)
        return
    if args.mode == "eval":
        _run_eval(cfg)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

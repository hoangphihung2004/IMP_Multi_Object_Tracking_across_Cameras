"""
Model Conversion Utility for Jetson DeepStream Pipeline.

Handles the full model pipeline:
    PTH (PyTorch) → ONNX → TensorRT Engine (.engine)

Usage:
    - As a module:  from core.model_converter import check_and_convert_models
    - As a script:  python -m core.model_converter

Accepts YOLOXConfig and SystemConfig Pydantic models for typed control,
with fallback to yolox_config.json → env vars → hardcoded defaults.
"""

import os
import shutil
import subprocess
import sys
from typing import Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────
# Defaults (only used when no config object / env / JSON available)
# ──────────────────────────────────────────────────────────────
DEFAULT_ENGINE_PATH = "pretrained/model_trt.engine"
DEFAULT_ONNX_PATH   = "pretrained/model_trt.onnx"
DEFAULT_PTH_PATH    = "pretrained/ocsort_x_mot20.pth.tar"
DEFAULT_EXP_FILE    = "exps/yolox_x_mix_mot20_ch.py"    
YOLOX_CONFIG_PATH   = "config/yolox_config.json"

TRTEXEC_CANDIDATES = [
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/tensorrt/bin/trtexec",
    shutil.which("trtexec") or "",
]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _load_yolox_config_json() -> dict:
    """Load paths from yolox_config.json if it exists (fallback only)."""
    import json
    if os.path.isfile(YOLOX_CONFIG_PATH):
        try:
            with open(YOLOX_CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            for key in ("model_path", "engine_path"):
                if key in cfg and cfg[key].startswith("./"):
                    cfg[key] = cfg[key][2:]
            return cfg
        except Exception as e:
            logger.warning(f"[ModelConverter] Could not read {YOLOX_CONFIG_PATH}: {e}")
    return {}


def _strip_dot_slash(path: str) -> str:
    """Remove leading ./ for consistency."""
    if path.startswith("./"):
        return path[2:]
    return path


def _find_trtexec() -> Optional[str]:
    """Locate the trtexec binary on the system."""
    for path in TRTEXEC_CANDIDATES:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _run(cmd: list, description: str) -> bool:
    """Run a subprocess with logging."""
    logger.info(f"[ModelConverter] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        logger.success(f"[ModelConverter] ✅ {description} succeeded.")
        return True
    except FileNotFoundError:
        logger.error(f"[ModelConverter] ❌ Command not found: {cmd[0]}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"[ModelConverter] ❌ {description} failed (exit code {e.returncode}).")
        return False


# ──────────────────────────────────────────────────────────────
# Stage 1: PTH → ONNX
# ──────────────────────────────────────────────────────────────
def convert_pth_to_onnx(
    pth_path: str,
    onnx_path: str,
    config_path: str = YOLOX_CONFIG_PATH,
    exp_file: str = None,
    opset: int = 18,
) -> bool:
    """
    Convert a PyTorch checkpoint to ONNX.

    Preferred: uses --config (yolox_config.json) to get architecture params.
    Fallback:  uses -f (exp_file) for legacy compatibility.
    """
    if not os.path.isfile(pth_path):
        logger.error(f"[ModelConverter] PTH checkpoint not found: {pth_path}")
        return False

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    cmd = [
        sys.executable,
        "deploy/scripts/export_onnx.py",
        "--output-name", onnx_path,
        "-c", pth_path,
        "-o", str(opset),
    ]

    # Prefer config mode over exp_file mode
    if os.path.isfile(config_path):
        cmd.extend(["--config", config_path])
        logger.info(f"[ModelConverter] Using config mode: {config_path}")
    elif exp_file and os.path.isfile(exp_file):
        cmd.extend(["-f", exp_file])
        logger.info(f"[ModelConverter] Using legacy exp_file mode: {exp_file}")
    else:
        logger.error(f"[ModelConverter] Neither config ({config_path}) nor exp_file found.")
        return False

    return _run(cmd, f"PTH → ONNX ({os.path.basename(onnx_path)})")


# ──────────────────────────────────────────────────────────────
# Stage 2: ONNX → TensorRT Engine
# ──────────────────────────────────────────────────────────────
def convert_onnx_to_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_mb: int = 7168,
) -> bool:
    """Build a TensorRT engine from an ONNX model using trtexec."""
    trtexec = _find_trtexec()
    if trtexec is None:
        logger.error(
            "[ModelConverter] trtexec not found! "
            "Ensure TensorRT is installed. Searched: "
            + ", ".join(p for p in TRTEXEC_CANDIDATES if p)
        )
        return False

    if not os.path.isfile(onnx_path):
        logger.error(f"[ModelConverter] ONNX model not found: {onnx_path}")
        return False

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")

    logger.info(f"[ModelConverter] Building TensorRT engine (FP16={fp16}, workspace={workspace_mb}MB)...")
    logger.info("[ModelConverter] ⏳ This may take 10-30 minutes on Jetson...")

    return _run(cmd, f"ONNX → TensorRT Engine ({os.path.basename(engine_path)})")


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────
def check_and_convert_models(yolox_cfg=None, system_cfg=None) -> None:
    """
    Full auto-conversion pipeline.

    Priority for resolving paths:
        YOLOXConfig object → ENV vars → yolox_config.json → hardcoded defaults

    Args:
        yolox_cfg: YOLOXConfig Pydantic model (provides engine_path, model_path).
        system_cfg: SystemConfig Pydantic model (provides fp16 flag).
    """
    # ── Resolve paths ────────────────────────────────────────
    if yolox_cfg is not None:
        _engine = _strip_dot_slash(yolox_cfg.engine_path)
        _pth    = _strip_dot_slash(yolox_cfg.model_path)
    else:
        _json = _load_yolox_config_json()
        _engine = _json.get("engine_path", DEFAULT_ENGINE_PATH)
        _pth    = _json.get("model_path", DEFAULT_PTH_PATH)

    engine_path = os.getenv("MODEL_ENGINE_PATH") or _engine
    pth_path    = os.getenv("MODEL_PTH_PATH")    or _pth
    onnx_path   = os.getenv("MODEL_ONNX_PATH",   DEFAULT_ONNX_PATH)
    exp_file    = os.getenv("MODEL_EXP_FILE",     DEFAULT_EXP_FILE)

    # FP16: SystemConfig → env → default True
    if system_cfg is not None:
        fp16 = system_cfg.fp16
    else:
        fp16 = os.getenv("TRT_FP16", "1") == "1"

    workspace = int(os.getenv("TRT_WORKSPACE_MB", "7168"))

    logger.info("=" * 60)
    logger.info("[ModelConverter] Checking model files...")
    logger.info(f"  Engine : {engine_path}")
    logger.info(f"  ONNX   : {onnx_path}")
    logger.info(f"  PTH    : {pth_path}")
    logger.info(f"  Exp    : {exp_file}")
    logger.info(f"  FP16   : {fp16}")
    logger.info("=" * 60)

    # ── Step 1: Engine exists → done ─────────────────────────
    if os.path.isfile(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        logger.success(f"[ModelConverter] ✅ TensorRT engine found ({size_mb:.1f} MB). Ready!")
        return

    logger.warning(f"[ModelConverter] Engine not found: {engine_path}")

    # ── Step 2: ONNX → Engine ────────────────────────────────
    if os.path.isfile(onnx_path):
        logger.info(f"[ModelConverter] ONNX found: {onnx_path}")
        if convert_onnx_to_engine(onnx_path, engine_path, fp16=fp16, workspace_mb=workspace):
            return
        logger.error("[ModelConverter] ❌ Engine build failed.")
        sys.exit(1)

    logger.warning(f"[ModelConverter] ONNX not found: {onnx_path}")

    # ── Step 3: PTH → ONNX → Engine ─────────────────────────
    if os.path.isfile(pth_path):
        logger.info("[ModelConverter] PTH found — starting full conversion.")
        if not convert_pth_to_onnx(pth_path, onnx_path):
            logger.error("[ModelConverter] ❌ PTH → ONNX failed.")
            sys.exit(1)
        if not convert_onnx_to_engine(onnx_path, engine_path, fp16=fp16, workspace_mb=workspace):
            logger.error("[ModelConverter] ❌ ONNX → Engine failed.")
            sys.exit(1)
        return

    # ── Step 4: Nothing ──────────────────────────────────────
    logger.error("=" * 60)
    logger.error("[ModelConverter] ❌ NO MODEL FILES FOUND!")
    logger.error(f"  engine: {engine_path} | onnx: {onnx_path} | pth: {pth_path}")
    logger.error("  Provide: .engine, .onnx, or .pth.tar + yolox_config.json")
    logger.error("=" * 60)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_and_convert_models()

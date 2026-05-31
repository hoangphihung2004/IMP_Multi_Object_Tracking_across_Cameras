#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX ONNX Export Script.

Supports TWO modes:
  1. Exp-file mode (legacy):  -f exps/yolox_x_mix_mot20_ch.py -c checkpoint.pth.tar
  2. Config mode (preferred):  --config config/yolox_config.json -c checkpoint.pth.tar

Config mode reads architecture params (num_classes, depth, width, tsize) from
yolox_config.json and builds the YOLOX model directly — no exp file needed.
"""

import argparse
import json
import os
import sys

from loguru import logger

import torch
from torch import nn

from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


# ──────────────────────────────────────────────────────────────
# Model Builder (replaces exp_file dependency)
# ──────────────────────────────────────────────────────────────
def build_yolox_model(num_classes: int, depth: float, width: float):
    """
    Build a YOLOX model from architecture params only.
    This replaces the need for an exp_file.
    """
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    backbone = YOLOPAFPN(depth=depth, width=width)
    head = YOLOXHead(num_classes=num_classes, width=width, in_channels=[256, 512, 1024])
    model = YOLOX(backbone, head)
    return model


# ──────────────────────────────────────────────────────────────
# Normalization Wrapper (bakes ImageNet mean/std into ONNX)
# ──────────────────────────────────────────────────────────────
class NormalizeWrapper(nn.Module):
    """
    Wraps a YOLOX model to prepend ImageNet normalization.
    Input:  [0, 1] scaled RGB tensor (what DeepStream sends with net-scale-factor=1/255)
    Output: normalized tensor matching Python preproc exactly.
    
    This eliminates the per-channel scale issue in nvinfer config.
    """
    def __init__(self, model: nn.Module, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


# ──────────────────────────────────────────────────────────────
# Core Export Logic
# ──────────────────────────────────────────────────────────────
def export_onnx(
    model: nn.Module,
    input_size: tuple,
    onnx_path: str,
    opset: int = 18,
    input_name: str = "images",
    output_name: str = "output",
    simplify_model: bool = True,
    rgb_means: tuple = (0.485, 0.456, 0.406),
    rgb_std: tuple = (0.229, 0.224, 0.225),
) -> None:
    """Export a PyTorch model to ONNX format with baked-in normalization."""

    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    model.eval()

    # Wrap model with normalization so DeepStream only needs to send [0,1] scaled pixels
    wrapped = NormalizeWrapper(model, rgb_means, rgb_std)
    wrapped.eval()

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    logger.info(f"Exporting ONNX (opset={opset}, input={input_size}, normalized=True)...")
    torch.onnx.export(
        wrapped,
        dummy_input,
        onnx_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
    )
    logger.success(f"ONNX export → {onnx_path}")

    if simplify_model:
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(onnx_path)
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, onnx_path)
            logger.success("ONNX model simplified successfully.")
        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification.")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")


# ──────────────────────────────────────────────────────────────
# Mode 1: Config-based Export (preferred)
# ──────────────────────────────────────────────────────────────
def export_from_config(
    config_path: str,
    ckpt_path: str,
    onnx_path: str,
    opset: int = 18,
) -> None:
    """
    Export ONNX using yolox_config.json instead of exp_file.
    Reads: num_classes, depth, width, tsize from config.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    num_classes = cfg.get("num_classes", 1)
    depth = cfg.get("depth", 1.33)
    width = cfg.get("width", 1.25)
    tsize = cfg.get("tsize", 640)
    input_size = (tsize, tsize)

    logger.info(f"[Config Mode] num_classes={num_classes}, depth={depth}, width={width}, tsize={tsize}")

    # Build model from params
    model = build_yolox_model(num_classes, depth, width)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Export
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    export_onnx(model, input_size, onnx_path, opset=opset)


# ──────────────────────────────────────────────────────────────
# Mode 2: Exp-file Export (legacy, backward compatible)
# ──────────────────────────────────────────────────────────────
def export_from_exp(
    exp_file: str,
    ckpt_path: str,
    onnx_path: str,
    opset: int = 18,
) -> None:
    """
    Export ONNX using the legacy exp_file approach.
    """
    from yolox.exp import get_exp

    exp = get_exp(exp_file, None)
    model = exp.get_model()

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    input_size = exp.test_size
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    export_onnx(model, input_size, onnx_path, opset=opset)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def make_parser():
    parser = argparse.ArgumentParser("YOLOX ONNX Export")
    parser.add_argument("--output-name", type=str, default="yolox.onnx", help="Output ONNX path")
    parser.add_argument("-c", "--ckpt", required=True, type=str, help="Checkpoint (.pth.tar) path")
    parser.add_argument("-o", "--opset", default=18, type=int, help="ONNX opset version")

    # Mode selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="[PREFERRED] Path to yolox_config.json")
    group.add_argument("-f", "--exp_file", type=str, help="[LEGACY] Path to experiment .py file")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info(f"Args: {args}")

    if args.config:
        logger.info(f"Using CONFIG mode: {args.config}")
        export_from_config(
            config_path=args.config,
            ckpt_path=args.ckpt,
            onnx_path=args.output_name,
            opset=args.opset,
        )
    else:
        logger.info(f"Using EXP-FILE mode: {args.exp_file}")
        export_from_exp(
            exp_file=args.exp_file,
            ckpt_path=args.ckpt,
            onnx_path=args.output_name,
            opset=args.opset,
        )


if __name__ == "__main__":
    main()

from loguru import logger

import tensorrt as trt
import torch
from torch2trt import TRTModule

from yolox.exp import get_exp

import argparse
import os
import shutil

def make_parser():
    parser = argparse.ArgumentParser("YOLOX TRT Packager")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("-w", "--workspace", type=int, default=1024, help="max workspace size in MB")
    
    # New argument to point to the existing engine file
    parser.add_argument("--engine", type=str, default="pretrained/ocsort_x_mot20.trt", help="path to existing TensorRT engine file from trtexec")
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    
    # Path to save the wrapped .pth model
    pth_output = os.path.join(file_name, "model_trt.pth")
    
    engine_path = args.engine
    
    if not os.path.exists(engine_path):
        logger.error(f"Engine file not found: {engine_path}")
        logger.info("Please run trtexec to generate the engine first!")
        return

    logger.info(f"Loading engine from {engine_path}...")
    
    # Load the engine manually
    with open(engine_path, "rb") as f:
        engine_data = f.read()

    # Create TRT Runtime and deserialize engine
    logger.info("Deserializing engine...")
    trt_logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        logger.error("Failed to deserialize engine!")
        return

    # Wrap in TRTModule
    logger.info("Wrapping in TRTModule...")
    # NOTE: input_names and output_names must match what trtexec/onnx produced
    model_trt = TRTModule(engine, input_names=["images"], output_names=["output"])
    
    # Save as .pth for demo_track.py
    torch.save(model_trt.state_dict(), pth_output)
    logger.info(f"Successfully packaged engine into {pth_output}")
    
    # Also save .engine file for consistency
    engine_file = os.path.join(file_name, "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(engine_data)
    
    logger.info("Done! Now you can run demo_track.py --trt")

if __name__ == "__main__":
    main()

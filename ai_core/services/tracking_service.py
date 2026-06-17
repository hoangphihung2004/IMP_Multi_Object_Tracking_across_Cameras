import cv2
import numpy as np
from typing import List, Any
import os.path as osp
import cv2
import torch
from loguru import logger
from exps.yolox_x_mix_mot20_ch import Exp
from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cuda"),
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)

            # Handle TensorRT output list/tuple
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Debug: print shape to understand structure
            # logger.info(f"Raw output shape: {outputs.shape}")

            # If outputs has 4 dims (1, 1, N, C), squeeze the second dim
            if outputs.dim() == 4 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)

            # Additional check: ensure it is (Batch, N, C)
            # If input batch is 1, output should be (1, N, C)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


class YOLOX:
    def __init__(self, args):
        self.args = args
        self.exp = Exp()
        self.setup_parameters()
        self.model = self.exp.get_model().to(self.args.device)
        logger.info("Model Summary: {}".format(get_model_info(self.model, self.exp.test_size)))
        self.model.eval()
        self.decoder = None

        self.setup_model()

    def setup_model(self):
        if not self.args.trt:  # nếu không dùng TensorRT, load checkpoint bình thường
            ckpt_file = self.args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            self.model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if self.args.fuse:
            logger.info("\tFusing model...")
            self.model = fuse_model(self.model)

        if self.args.fp16:
            self.model = self.model.half()  # to FP16

        if self.args.trt:
            assert not self.args.fuse, "TensorRT model is not support model fusing!"
            trt_file = self.args.trt_file
            assert osp.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            self.model.head.decode_in_inference = False
            self.decoder = self.model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            self.args.trt_file = None
            self.decoder = None

    def setup_parameters(self):
        if self.args.trt:
            self.args.device = "gpu"
        self.args.device = torch.device("cuda" if self.args.device == "gpu" else "cpu")

        if self.args.conf is not None:
            self.exp.test_conf = self.args.conf
        if self.args.nms is not None:
            self.exp.nmsthre = self.args.nms
        if self.args.tsize is not None:
            self.exp.test_size = (self.args.tsize, self.args.tsize)

        logger.info("Args: {}".format(self.args))


class TrackingService:
    def __init__(self, args):

        self.yolo = YOLOX(args=args)

        self.predictor = Predictor(self.yolo.model,
                                   self.yolo.exp,
                                   self.yolo.args.trt_file,
                                   self.yolo.decoder,
                                   self.yolo.args.device,
                                   self.yolo.args.fp16)

        self.tracker = OCSort(det_thresh=self.yolo.args.track_thresh,
                              max_age=self.yolo.args.max_age,
                              min_hits=self.yolo.args.min_hits,
                              iou_threshold=self.yolo.args.iou_thresh,
                              delta_t=self.yolo.args.delta_t,
                              asso_func=self.yolo.args.asso_func,
                              inertia=self.yolo.args.inertia,
                              use_byte=self.yolo.args.use_byte)

        self.timer = Timer()

    def predict(self, frame_id, frame, box_type="xyxy"):
        bboxes, track_ids, _, avg_time = self.predict_with_scores(frame_id, frame, box_type)
        return bboxes, track_ids, avg_time

    def predict_with_scores(self, frame_id, frame, box_type="xyxy"):

        if frame_id % 30 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, self.timer.average_time)))

        outputs, img_info = self.predictor.inference(frame, self.timer)
        bboxes = []
        track_ids = []
        scores = []

        if outputs[0] is not None:

            targets = self.tracker.update(output_results=outputs[0],
                                         img_info=[img_info['height'],
                                                   img_info['width']],
                                         img_size=self.yolo.exp.test_size)

            for t in targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                xyxy = [t[0], t[1], t[2], t[3]]
                tid = t[4]
                score = float(t[5]) if len(t) > 5 else 1.0
                vertical = tlwh[2] / tlwh[3] > self.yolo.args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.yolo.args.min_box_area and not vertical:
                    bboxes.append(xyxy if box_type == "xyxy" else tlwh)
                    track_ids.append(tid)
                    scores.append(score)
            self.timer.toc()

        return bboxes, track_ids, scores, self.timer.average_time

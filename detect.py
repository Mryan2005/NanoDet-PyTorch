import cv2
import os
import time
import torch
import argparse
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
import numpy as np
import cv2
import copy
import pycocotools.mask as mask_util
import matplotlib as mpl
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def detect(cfgFile="./config/nanodet-m.yml", modelFile="model/nanodet_m.pth", forceUseCPU=False):
    load_config(cfg, cfgFile)
    logger = Logger(-1, use_tensorboard=False)
    if torch.cuda.is_available() and not forceUseCPU:
        device = 'cuda:0'
    else:
        device = 'cpu'
    predictor = Predictor(cfg, modelFile, logger, device)
    return predictor


def predict(predictor, img, score_thresh=0.35):
    meta, res = predictor.inference(img)
    all_box = []
    dets = res
    class_names = cfg.class_names
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    resBox = []
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        text = '{}'.format(class_names[label])
        resBox.append((text, x0, y0, x1, y1, score))
    return resBox


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    predictor = detect(forceUseCPU=True)
    cap = cv2.VideoCapture("1.mp4")
    while True:
        start = time.time()
        ret_val, frame = cap.read()
        if not ret_val:
            break
        resBox = predict(predictor, frame, score_thresh=0.35)
        for box in resBox:
            text, x0, y0, x1, y1, score = box
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        print("time:", time.time() - start)
        print('FPS:', 1 / (time.time() - start))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

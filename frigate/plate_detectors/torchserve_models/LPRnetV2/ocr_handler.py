import base64
import io
import logging
import os
from typing import Sequence
import cv2
import numpy as np
import torch
import yaml
from argparse import Namespace
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image

class ModelHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        with open(os.path.join(model_dir, 'config.yaml')) as f:
            self.args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        logging.info(self.device)

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            width = int.from_bytes(row.get("width", 0), 'big')
            height = int.from_bytes(row.get("height", 0), 'big')
            dtype_str = row.get("dtype").decode("ascii")
            if isinstance(image, str):
                image = base64.b64decode(image)
            if isinstance(image, (bytearray, bytes)):
                image = np.frombuffer(image, dtype=np.dtype(getattr(np, dtype_str))).reshape([height, width, 3])
                image = self.numpy2tensor(image, self.args.img_size)
            else:
                image = torch.FloatTensor(image)
            images.append(image)
        return torch.stack(images)

    def postprocess(self, logit):
        logit = logit.detach().to('cpu')
        pred, _ = self.decode(logit, self.args.chars)
        return pred

    @staticmethod
    def numpy2tensor(img: np.ndarray, img_size: Sequence[int]):
        # convert a numpy image to tensor
        height, width, _ = img.shape

        if height != img_size[1] or width != img_size[0]:
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img)

    @staticmethod
    def decode(preds, chars):
        # greedy decode
        pred_labels = list()
        labels = list()
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = list()
            for j in range(pred.shape[1]):
                pred_label.append(np.argmax(pred[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = ''
            for c in pred_label:  # dropout repeated label and blank label
                if (pre_c == c) or (c == len(chars) - 1):
                    if c == len(chars) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            pred_labels.append(no_repeat_blank_label)

        for i, label in enumerate(pred_labels):
            lb = ""
            for i in label:
                lb += chars[i]
            labels.append(lb)

        return labels, pred_labels
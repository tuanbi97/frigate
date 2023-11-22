import base64
import importlib
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from detect_config import cfg_plate
from ts.torch_handler.base_handler import BaseHandler, XLA_AVAILABLE
from ts.utils.util import list_classes_from_module
from box_utils import decode, decode_landm
from py_cpu_nms import py_cpu_nms
from prior_box import PriorBox
from PIL import Image


class ModelHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        self.config = cfg_plate
        self.image_size = (cfg_plate['max_size'],
                           cfg_plate['max_size'] * (
                                   cfg_plate['image_ratio'][1] / cfg_plate['image_ratio'][0]))  # width, height
        priorbox = PriorBox(self.config, image_size=(self.image_size[1], self.image_size[0]), phase='test')  # height, width
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        def remove_prefix(state_dict, prefix):
            ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
            print('remove prefix \'{}\''.format(prefix))
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class()
        if model_pt_path:
            map_location = (
                None if (XLA_AVAILABLE and self.map_location is None) else self.device
            )
            state_dict = torch.load(model_pt_path, map_location=map_location)
            if "state_dict" in state_dict.keys():
                state_dict = remove_prefix(state_dict['state_dict'], 'module.')
            else:
                state_dict = remove_prefix(state_dict, 'module.')
            model.load_state_dict(state_dict)
        return model

    def preprocess(self, data):
        images = []
        image_shapes = []
        for row in data:
            image = row.get("data") or row.get("body")
            width = int.from_bytes(row.get("width", 0), 'big')
            height = int.from_bytes(row.get("height", 0), 'big')
            dtype_str = row.get("dtype").decode("ascii")
            if isinstance(image, str):
                image = base64.b64decode(image)
                image_shape = np.shape(image)
            if isinstance(image, (bytearray, bytes)):
                image = np.frombuffer(image, dtype=np.dtype(getattr(np, dtype_str))).reshape([height, width, 3])
                image, image_shape = self.preprocess_one_image(image, width, height)
            else:
                image = torch.FloatTensor(image)
                image_shape = image.shape
            images.append(image)
            image_shapes.append(image_shape)
        return torch.stack(images), image_shapes

    def preprocess_one_image(self, img, w, h):
        img_max_size = max(h, w)
        img = np.pad(
            img,
            ((0, img_max_size - h), (0, img_max_size - w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        max_size = self.config['max_size']
        im_shape = img.shape
        im_size_max = np.max(im_shape[0:2])
        resize = float(max_size) / float(im_size_max)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), im_shape

    def inference(self, data, *args, **kwargs):
        inputs = data[0]
        img_shapes = data[1]
        with torch.no_grad():
            marshalled_data = inputs.to(self.device)
            results = self.model(marshalled_data, *args, **kwargs)
        return results[0], results[1], results[2], img_shapes

    def postprocess(self, data):
        batch_detection_boxes, batch_detection_scores, batch_detection_landmark, img_shapes = data
        results = []
        for detection_boxes, detection_scores, detection_landmark, img_shape in zip(batch_detection_boxes, batch_detection_scores, batch_detection_landmark, img_shapes):
            scores = F.softmax(detection_scores, dim=-1)
            boxes = decode(detection_boxes, self.prior_data, self.config['variance'])
            boxes[:, 0::2] = boxes[:, 0::2] * img_shape[1]  # width
            boxes[:, 1::2] = boxes[:, 1::2] * img_shape[0]  # height

            landms = decode_landm(detection_landmark, self.prior_data, self.config['variance'])
            landms[:, 0::2] = landms[:, 0::2] * img_shape[1]
            landms[:, 1::2] = landms[:, 1::2] * img_shape[0]

            if self.device != 'cpu':
                scores = scores.cpu().detach().numpy()
                boxes = boxes.cpu().detach().numpy()
                landms = landms.cpu().detach().numpy()
            else:
                scores = scores.detach().numpy()
                boxes = boxes.detach().numpy()
                landms = landms.detach().numpy()
            scores = scores[:, 1]
            # ignore low scores
            inds = np.where(scores > self.config['confidence_threshold'])[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.config['top_k']]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.config['nms_threshold'])
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:self.config['keep_top_k'], :]
            landms = landms[:self.config['keep_top_k'], :]

            dets = np.concatenate((dets, landms), axis=1)
            results.append(dets.tolist())
        return results
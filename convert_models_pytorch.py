import os
import shutil
import tempfile
import cv2

import tensorflow as tf

from frigate.plate_detectors.alpr.plate_detector_gpu import Plate_Detector

# onnx - step 1
from torch.autograd import Variable
import torch.onnx

# onnx - step 2
import onnx
from onnx_tf.backend import prepare

tmp_path = tempfile.mkdtemp()

image = cv2.imread("/workspace/frigate/frigate/plate_detectors/alpr/test.jpg")
detector = Plate_Detector()
# image_processed = detector.get_input(image)
# print(image_processed.shape)
dummy_input = Variable(torch.randn(1, 3, 650, 650)).to(detector.device)
onnx_model_path = os.path.join(tmp_path, 'LP_detect_92.onnx')
torch.onnx.export(detector.model, dummy_input, onnx_model_path, opset_version=12,
                  input_names = ['input'],   # the model's input names
                  output_names = ['loc', 'conf', 'landms'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size', 2: 'width', 3: 'height'},    # variable length axes
                                'loc' : {0 : 'batch_size'},
                                'conf': {0 : 'batch_size'},
                                'landms': {0: 'batch_size'}})

onnx_model = onnx.load(onnx_model_path)
opset_version = onnx_model.opset_import[0].version if len(onnx_model.opset_import) > 0 else None
print(opset_version)


tf_rep = prepare(onnx_model, strict=False)
tf_pb_path = '/workspace/frigate/frigate/plate_detectors/alpr/models/retina_plate/1'
tf_rep.export_graph(tf_pb_path)
shutil.rmtree(tmp_path)

import time
from multiprocessing import Process

import cv2
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

# def detect(channel):
#     # with tf.device('/gpu:0'):
#     start = time.time()
#     request = predict_pb2.PredictRequest()
#     stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
#     request.model_spec.name = "plate_recognition"
#     request.model_spec.signature_name = "classification"
#     img = cv2.imread("/workspace/frigate/frigate/plate_detectors/alpr/test.jpg")
#     img = cv2.resize(img, (92, 24))
#     img = np.expand_dims(img, axis=0)
#     img = np.array(img, dtype=np.float32)
#     request.inputs["inputs"].CopyFrom(tf.make_tensor_proto(img[0], shape=img.shape))
#     stub.Predict(request)
#     print(time.time() - start)


def detect(channel):
    # with tf.device('/gpu:0'):
    start = time.time()
    request = predict_pb2.PredictRequest()
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request.model_spec.name = "plate_detection"
    request.model_spec.signature_name = "serving_default"
    img = cv2.imread("/workspace/frigate/frigate/plate_detectors/alpr/test.jpg")
    max_size = 650
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(max_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if resize != 1:
        img = cv2.resize(
            img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
        )
    # img = cv2.resize(
    #         img, None, None, fx=max_size/im_shape[1], fy=max_size/im_shape[0], interpolation=cv2.INTER_LINEAR
    #     )
    img = np.float32(img)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    request.inputs["input"].CopyFrom(tf.make_tensor_proto(img[0], shape=img.shape))
    out = stub.Predict(request)
    print(len(out.outputs["loc"].float_val))
    print(out.outputs["loc"].tensor_shape)
    tf.make_ndarray(out.outputs["loc"])
    # print(output1)
    print(len(out.outputs["conf"].float_val))
    print(out.outputs["conf"].tensor_shape)
    print(len(out.outputs["landms"].float_val))
    print(out.outputs["landms"].tensor_shape)
    # print(out.outputs['575'])
    # print(out.outputs['output'])
    print(time.time() - start)


hostport = "localhost:8500"
channel = grpc.insecure_channel(hostport)
for i in range(0, 1):
    p = Process(target=detect, args=(channel,))
    p.start()
    p.join()

# for i in range(0, 10000):
#     p = Process(target=detect, args=(channel,))
#     p.start()

import cv2
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from multiprocessing import Process

def detect(channel):
    with tf.device('/gpu:0'):
        request = predict_pb2.PredictRequest()
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request.model_spec.name = 'plate_recognition'
        request.model_spec.signature_name = 'classification'
        img = cv2.imread('/workspace/frigate/frigate/plate_detectors/alpr/test.jpg')
        img = cv2.resize(img, (92, 24))
        img = np.expand_dims(img, axis=0)
        img = np.array(img, dtype=np.float32)
        request.inputs['inputs'].CopyFrom(
            tf.make_tensor_proto(img[0], shape=img.shape))
        result = stub.Predict(request)
        print(result)

hostport = 'localhost:8500'
channel = grpc.insecure_channel(hostport)
for i in range(0, 1000):
    p = Process(target=detect, args=(channel,))
    p.start()
    p.join()

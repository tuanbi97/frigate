import logging
import sys
import traceback

import tensorflow as tf
import torch
from tensorflow_serving.apis import predict_pb2

from frigate.plate_detectors.alpr.plate_checker import Plate_Checker
from frigate.plate_detectors.alpr.plate_detector_gpu import Plate_Detector
from frigate.plate_detectors.alpr.plate_recognizer import Plate_Recognizer
from frigate.plate_detectors.remote_plate_detector.remote_plate_detector import (
    RemotePlateDetector,
)


class AlprPlateDetector(RemotePlateDetector):
    def __init__(self, grpc_channel):
        super().__init__(grpc_channel)
        self.checker = Plate_Checker()

    def detect_plate(self, image):
        detector = Plate_Detector(use_remote_model=True)
        image_processed = detector.get_input(image)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = "plate_detection"
        request.model_spec.signature_name = "serving_default"
        request.inputs["input"].CopyFrom(
            tf.make_tensor_proto(image_processed[0], shape=image_processed.shape)
        )
        try:
            response = self.prediction_service_stub.Predict(request)
        except Exception as e:
            logging.warn(e)
            return []
        loc_array = tf.make_ndarray(response.outputs["loc"])
        conf_array = tf.make_ndarray(response.outputs["conf"])
        landms_array = tf.make_ndarray(response.outputs["landms"])
        loc = torch.from_numpy(loc_array).to(detector.device)
        conf = torch.from_numpy(conf_array).to(detector.device)
        landms_array = torch.from_numpy(landms_array).to(detector.device)
        try:
            detection_result = detector.post_process(loc, conf, landms_array)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logging.error(
                "Error: {}\nTraceback: {}\n".format(
                    e, repr(traceback.extract_tb(exc_traceback))
                )
            )
            logging.info(f"{loc.shape} : {detector.prior_data.shape}")
            logging.warn("Something wrong when post processing")
            return []
        return detection_result

    def recognize_plate(self, image):
        recognizer = Plate_Recognizer(use_remote_model=True)
        ocr_image_processed = recognizer.get_input(image)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = "plate_recognition"
        request.model_spec.signature_name = "classification"
        request.inputs["inputs"].CopyFrom(
            tf.make_tensor_proto(
                ocr_image_processed[0], shape=ocr_image_processed.shape
            )
        )
        try:
            response = self.prediction_service_stub.Predict(request)
        except Exception:
            return None
        ocr_model_out = tf.make_ndarray(response.outputs["outputs"])
        ocr_result = recognizer.post_process(ocr_model_out)
        plate_number = ocr_result[0]
        check_result = self.checker.run(plate_number)
        if not check_result or not check_result[1]:
            return None

        return plate_number

import tensorflow as tf
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
        self.plate_detector = Plate_Detector(
            self.prediction_service_stub, use_remote_model=True
        )

    def detect_plate(self, image):
        return self.plate_detector.detect(image)

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

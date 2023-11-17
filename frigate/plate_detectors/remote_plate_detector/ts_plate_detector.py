from frigate.plate_detectors.remote_plate_detector import inference_pb2, inference_pb2_grpc
from frigate.plate_detectors.alpr.plate_checker import Plate_Checker

from frigate.plate_detectors.remote_plate_detector.remote_plate_detector import RemotePlateDetector
import json

class TsPlateDetector(RemotePlateDetector):
    def __init__(self, grpc_channel):
        super().__init__(grpc_channel)
        self.prediction_service_stub = inference_pb2_grpc.InferenceAPIsServiceStub(grpc_channel)
        self.plate_detection_model_name = "retinanet"
        self.plate_recognition_model_name = "lprnet"
        self.checker = Plate_Checker()

    def detect_plate(self, image):
        h, w, _ = image.shape
        input_data = {"data": image.tobytes(),
                      "height": h.to_bytes(8, 'big'),
                      "width": w.to_bytes(8, 'big'),
                      "dtype": bytes(str(image.dtype), 'ascii')}
        response = self.prediction_service_stub.Predictions(
            inference_pb2.PredictionsRequest(model_name=self.plate_detection_model_name, input=input_data),
        )
        return json.loads(response.prediction.decode("utf-8"))

    def recognize_plate(self, image):
        h, w, _ = image.shape
        input_data = {"data": image.tobytes(),
                      "height": h.to_bytes(8, 'big'),
                      "width": w.to_bytes(8, 'big'),
                      "dtype": bytes(str(image.dtype), 'ascii')}
        response = self.prediction_service_stub.Predictions(
            inference_pb2.PredictionsRequest(model_name=self.plate_recognition_model_name, input=input_data),
        )
        plate_number = response.prediction.decode("utf-8")
        check_result = self.checker.run(plate_number)
        if not check_result or not check_result[1]:
            return None

        return plate_number
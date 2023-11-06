from abc import ABC, abstractmethod
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


class RemotePlateDetector(ABC):
    def __init__(self, grpc_channel) -> None:
        super().__init__()
        self.grpc_channel = grpc_channel
        self.prediction_service_stub = prediction_service_pb2_grpc.PredictionServiceStub(grpc_channel)

    @abstractmethod
    def detect_plate(self, image):
        raise NotImplementedError("Must override detect_plate")

    @abstractmethod
    def recognize_plate(self, image):
        raise NotImplementedError("Must override recognize_plate")

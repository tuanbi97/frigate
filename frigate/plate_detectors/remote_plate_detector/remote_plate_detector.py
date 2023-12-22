from abc import ABC, abstractmethod


class RemotePlateDetector(ABC):
    def __init__(self, grpc_channel) -> None:
        super().__init__()
        self.grpc_channel = grpc_channel

    @abstractmethod
    def detect_plate(self, image):
        raise NotImplementedError("Must override detect_plate")

    @abstractmethod
    def recognize_plate(self, image):
        raise NotImplementedError("Must override recognize_plate")

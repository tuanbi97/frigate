import cv2
import grpc
import numpy as np

from frigate.plate_detectors.remote_plate_detector.alpr_plate_detector import AlprPlateDetector
from frigate.plate_detectors.alpr.retina_plate.utils.utils import img_transform

hostport = "localhost:8500"
channel = grpc.insecure_channel(hostport)
detector = AlprPlateDetector(channel)
img = cv2.imread("/workspace/frigate/frigate/plate_detectors/alpr/test.jpg")
detection_result = detector.detect_plate(img)
for i, b in enumerate(detection_result):
    _, plate = img_transform(img, detection_result[i][5:])
    ans = detector.recognize_plate(np.array(plate, dtype=np.float32))
    print(ans)
channel.close()
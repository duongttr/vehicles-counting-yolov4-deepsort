from VehicleDetection import VehicleDetection
import cv2
import numpy as np

settings = {
    'labels': 'yolo/obj.names',
    'cfg': 'yolo/yolov4-custom.cfg',
    'weight': 'yolo/ylov4_final.weights'
}

vd = VehicleDetection(settings)

# Input your image path
img_path = './test.jpeg'

(img_new, _, _, _) = vd1.detect_image(img_path,
                is_returned_image=True)

cv2.imshow('result',  img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

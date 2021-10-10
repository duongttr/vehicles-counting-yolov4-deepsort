import numpy as np
import time
import cv2
from FormatConverter import LTWH2YOLOv4Format as converter
import os

class VehicleDetection:
    def __init__(self, yolo):
        self.LABELS = open(yolo['labels']).read().strip().split('\n')
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))
        self.net = cv2.dnn.readNetFromDarknet(yolo['cfg'], yolo['weight'])

    def detect_image(self, image_path, confidence=0.5, threshold=0.3, is_returned_image=False):
        """
        - confidence: Minimum probability to filter weak detections. I’ve given this a default value of 50%,
                but you should feel free to experiment with this value.
        - threshold: This is our non-maxima suppression threshold with a default value of 0.3
        """
        img = cv2.imread(image_path)
        (H, W) = img.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()

        print("[INFO] YOLO took: {:.6f} seconds for {}".format(end - start, image_path))

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]

                if conf > confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")


                    # top-left coordinates
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x,y,int(width), int(height)])
                    confidences.append(float(conf))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)


        ret_bboxes = []
        ret_classes = []
        ret_confs = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                ret_bboxes += [converter([x,y,w,h], W,H)]
                ret_classes += [classIDs[i]]
                ret_confs += [confidences[i]]
                if is_returned_image:
                    color = [int(c) for c in self.COLORS[classIDs[i]]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        if is_returned_image:
            return (img, ret_classes, ret_bboxes, ret_confs)
        else:
            return (None, ret_classes, ret_bboxes, None)

import numpy as np
import time
import cv2
import os

class YOLO:
    def __init__(self, labels, cfg, weight, use_gpu=False):
        """
        Parameters:
        - labels: path to labels' file
        - cfg: path to config file
        - weight: path to weight model
        - use_gpu: enable this if your machine is supported (CUDA GPU only)
        """
        self.LABELS = open(labels).read().strip().split('\n')
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))
        self.net = cv2.dnn.readNetFromDarknet(cfg, weight)
        self.ln = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]
        # Some OpenCV versions return error at the above line, comment that line and use following
        # self.ln = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # To use gpu, your machine need to install OpenCV supporting GPU runtime
        # Reading this post for installing on Google Colab:
        # https://answers.opencv.org/question/233476/how-to-make-opencv-use-gpu-on-google-colab/
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    def detect_image(self, image_path, confidence=0.5, threshold=0.3, return_result_image=True):
        """
        Returns result info includes class id, bounding box and confidence of each object.
        Parameters:
        - image_path: path to image for detection
        - confidence: Minimum probability to filter weak detections. I’ve given this a default value of 50%,
                but you should feel free to experiment with this value.
        - threshold: This is our non-maxima suppression threshold with a default value of 0.3
        - return_result_image: Return result image
        """
        img = cv2.imread(image_path)
        (H, W) = img.shape[:2]


        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        self.net.setInput(blob)

        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()

        print("[INFO] YOLO took: {:.3f} seconds for {}".format(end - start, image_path))

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

        #
        # ret_bboxes = []
        # ret_classes = []
        # ret_confs = []
        result_info = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                result_info.append({'class_id': classIDs[i], 'bbox_2': [x,y,w,h], 'bbox': [(x/2+w/2)/W, (y/2+h/2)/H, w/W, h/H], 'conf': confidences[i]})
                if return_result_image:
                    color = [int(c) for c in self.COLORS[classIDs[i]]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        if return_result_image:
            return (img, result_info)
        else:
            return result_info

import cv2
import numpy as np

class RegionChooser:
    def __init__(self, frame):
        self._frame = frame
        self._clone = self._frame.copy()
        self._coords = []
        cv2.namedWindow("RegionChooser")
        cv2.setMouseCallback("RegionChooser", self._on_mouse)
    
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clone = self._frame.copy()
            self._coords += [(x,y)]
            print(self._coords)
        elif event == cv2.EVENT_LBUTTONUP:
            # draw to image
            self._shape = np.array([self._coords], dtype=np.int32)
            cv2.polylines(self._clone, pts=self._shape, isClosed=True, color=(55,243,46))
            cv2.imshow("RegionChooser", self._clone)
            

        elif event == cv2.EVENT_RBUTTONDOWN:
            # clear region
            self._clone = self._frame.copy()
            self._coords = []
            

    def set_shape(self, coords):
        self._shape = np.array([coords], dtype=np.int32)
    
    def cut(self, current_frame):
        mask = np.zeros(current_frame.shape, dtype=np.uint8)
        channel_count = current_frame.shape[2]
        ignore_mask_color = (0,) * channel_count
        cv2.fillPoly(mask, self._shape, ignore_mask_color)
        masked_image = cv2.bitwise_and(current_frame, mask)
        return masked_image

    def show(self):
        while True:
            cv2.imshow("RegionChooser", self._clone)
            key = cv2.waitKey(1)

            if key == ord('q'):
                cv2.destroyAllWindows()
                break
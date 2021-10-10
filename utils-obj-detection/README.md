# Testing your YOLOv4 model with OpenCV
- `AnnotationGenerator.py`: create annotation file automatically using a pre-trained model (This is useful when you want to train new data and you're too lazy to generate annotation file manually 😴).
```bash
python AnnotationGenerator.py -d <path-to-dataset-folder> -c <path-to-cfg-file> -w <path-to-weights-model> -lb <path-to-label-file>
```
- `FormatConverter.py`: function to convert coordinate's format in YOLOv4 and OpenCV.
- `SplitVideoIntoFrame.py`: split video into frame
```bash
python SplitVideoIntoFrame.py -v <path-to-video> -d <path-to-saved-directory> -e <image-extension> -fps <frames-per-second>
```
- `VehicleDetection.py`: mail file for processing detection.
- `test-obj-detection.py`: test your object detection.

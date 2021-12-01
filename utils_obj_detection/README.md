# Testing your YOLOv4 model with OpenCV
- `AnnotationGenerator.py`: create annotation file automatically using a pre-trained model (This is useful when you want to train new data and you're too lazy to generate annotation file manually 😴).
```bash
python AnnotationGenerator.py -d <path-to-dataset-folder> -c <path-to-cfg-file> -w <path-to-weights-model> -lb <path-to-label-file>
```
- `YOLO.py`: main file for processing detection.

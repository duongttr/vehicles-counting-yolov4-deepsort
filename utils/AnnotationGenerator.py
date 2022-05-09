from VehicleDetection import VehicleDetection

import argparse
import os
import glob

ap = argparse.ArgumentParser()

# python3 AnnotationGenerator.py -d MB10000 -c yolo/yolov4-custom.cfg -w yolo/yolov4-custom_final.weights -lb yolo/obj.names

ap.add_argument('-d', '--dataset', required=True,
                help='Path to dataset')
ap.add_argument('-c', '--cfg', required=True,
                help='Path to YOLO config file')
ap.add_argument('-w', '--weights', required=True,
                help='Path to YOLO pre-trained weights')
ap.add_argument('-lb', '--labels', required=True,
                help='Path to text file containing class names')
args = ap.parse_args()

yolo = {
    'labels': args.labels,
    'cfg': args.cfg,
    'weight': args.weights
}

vDetection = VehicleDetection(yolo)

IMAGE_EX = ['jpg', 'jpeg', 'png']
IMAGE_LIST = []
for ex in IMAGE_EX:
    IMAGE_LIST += glob.glob(os.path.join(args.dataset, f"*.{ex}"))

cnt = 0
len_img_list = len(IMAGE_LIST)

for IMAGE in IMAGE_LIST:
    if IMAGE.split('.')[-1] in ['jpeg', 'jpg', 'png']:
        (_, classes, boxes,_) = vDetection.detect_image(IMAGE)
        fname = IMAGE.rsplit('/', 1)[-1].split('.')[0]
        ANNO = os.path.join(args.dataset, fname + ".txt")
        with open(ANNO, 'w') as f:
            cnt += 1
            print(f'Generating {cnt}/{len_img_list}:  {ANNO}')
            for id, box in zip(classes, boxes):
                f.write(str(id + 4) + ' ' + ' '.join(map(str, box)) + '\n')
print(f'[INFO] Done! :)')

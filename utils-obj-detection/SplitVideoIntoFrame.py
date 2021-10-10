import cv2
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help='Path to video')
ap.add_argument('-d', '--directory', required=True,
                help='Path to saved directory')
ap.add_argument('-e', '--extension', default='.jpg',
                help='Image extension')
ap.add_argument('-fps', '--fps', default=3, type=int,
                help='Number of frames per second')
args = ap.parse_args()

vidcap = cv2.VideoCapture(args.video)
success,image = vidcap.read()

prefix_name = args.video.rsplit('/', 1)[-1].split('.')[0]
count = 0
while success:
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count * 1000 / args.fps))
  cv2.imwrite(os.path.join(args.directory, f"{prefix_name}_frame_{count}.{args.extension}"), image)     
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
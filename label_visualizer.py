import random
import json
import cv2
import sys


if len(sys.argv) not in [2,3]:
    print('Usage:')
    print('> python ./label_visulalizer.py <sample_path> [<output_path>]')
    print('Where <sample_path> is path to a sample without extension. The data will be infered by appending')
    print('a ".png" extension, and the label will be infered by appending a ".json" extension. If no <output_path>')
    print('is specified, the script will attempt to open a window to desplay the result. Otherwise, the result')
    print('will be written to output_path. (output_path should end in ".png" if present)')

# Load image and label
data_path  = f'{sys.argv[1]}.png'
label_path = f'{sys.argv[1]}.json'

captcha_image = cv2.imread(data_path)
with open(label_path) as f:
    label_obj = json.load(f)

print(captcha_image.shape)

def random_color():
    return ( random.choice(range(128, 256)), 
             random.choice(range(128, 256)),
             random.choice(range(128, 256)) )

for rect in label_obj:
    top_left     = (rect['left'] , rect['top'])
    bottom_right = (rect['right'], rect['bottom'])
    cv2.rectangle(captcha_image, top_left, bottom_right, random_color(), 3)

if len(sys.argv) == 3:
    output_path = sys.argv[2]
    cv2.imwrite(output_path, captcha_image)

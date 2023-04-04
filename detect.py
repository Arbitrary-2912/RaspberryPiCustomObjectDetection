"""
FRC: 2023 Game Element Custom Decision

Resources:
https://coral.ai/examples/
https://www.tensorflow.org/lite/examples

Author: The Green Machine - 2023
"""

import time
import numpy as np
from PIL import Image
import tensorflow as tf

import os
import cv2

cap = cv2.VideoCapture(0)

edgetpu = '0'  # make it '1' if Coral Accelerator is attached and use model with 'edgetpu' name

# Camera properties
horizontal_mount_offset = 0  # degrees
vertical_mount_offset = 0  # degrees

horizontal_FOV = 60  # degrees
vertical_FOV = 60  # degrees

# Model and Label Files

model_dir = os.path.join('models', 'custom')

model = 'frc2023elements.tflite'  # if not using edge tpu
# model = 'frc2023elements_edgetpu.tflite' # if using edge tpu

label = 'frc2023elements_labels.txt'

model_path = os.path.join(model_dir, model)
label_path = os.path.join(model_dir, label)


# -------------------Object Detection--------------------#
def detect_objects(interpreter, image, score_threshold=0.3, top_k=6):
    """Returns list of detected objects."""
    set_input_tensor(interpreter, image)
    # interpreter.invoke()
    invoke_interpreter(interpreter)

    global model_dir
    if (model_dir == os.path.join('models', 'pretrained')):
        # for pre-trained models
        boxes = get_output_tensor(interpreter, 0)
        class_ids = get_output_tensor(interpreter, 1)
        scores = get_output_tensor(interpreter, 2)
        count = int(get_output_tensor(interpreter, 3))
    else:
        # for custom models made by Model Maker
        scores = get_output_tensor(interpreter, 0)
        boxes = get_output_tensor(interpreter, 1)
        count = int(get_output_tensor(interpreter, 2))
        class_ids = get_output_tensor(interpreter, 3)

    def get_area(b):
        return abs(b.xmax - b.xmin) * abs(b.ymax - b.ymin)  # range: [0, 1]

    def get_center(b):
        return (float(b.xmin) + float(b.xmax)) / 2, (float(b.ymin) + float(b.ymax)) / 2  # x_range: [0, 1], y_range: [0, 1]

    def get_angles(b):
        p = get_center(b)

        px = p[0]
        py = p[1]

        nx = px - 0.5
        ny = py - 0.5

        vpw = 2 * np.tan(horizontal_FOV / 2.)  # visual plane width
        vph = 2 * np.tan(vertical_FOV / 2.)  # visual plane height

        x = vpw / 2 * nx
        y = vph / 2 * ny

        ax = np.arctan2(1, x)
        ay = np.arctan2(1, y)

        return float(ax + horizontal_mount_offset), float(ay + vertical_mount_offset)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)),
            area=get_area(
                BBox(xmin=np.maximum(0.0, xmin),
                     ymin=np.maximum(0.0, ymin),
                     xmax=np.minimum(1.0, xmax),
                     ymax=np.minimum(1.0, ymax))),
            center=Center(
                xcenter=get_center(
                    BBox(xmin=np.maximum(0.0, xmin),
                         ymin=np.maximum(0.0, ymin),
                         xmax=np.minimum(1.0, xmax),
                         ymax=np.minimum(1.0, ymax)))[0],
                ycenter=get_center(
                    BBox(xmin=np.maximum(0.0, xmin),
                         ymin=np.maximum(0.0, ymin),
                         xmax=np.minimum(1.0, xmax),
                         ymax=np.minimum(1.0, ymax)))[1]),
            angles=Angles(
                tx=get_angles(
                    BBox(xmin=np.maximum(0.0, xmin),
                         ymin=np.maximum(0.0, ymin),
                         xmax=np.minimum(1.0, xmax),
                         ymax=np.minimum(1.0, ymax))
                )[0],
                ty=get_angles(
                    BBox(xmin=np.maximum(0.0, xmin),
                         ymin=np.maximum(0.0, ymin),
                         xmax=np.minimum(1.0, xmax),
                         ymax=np.minimum(1.0, ymax))
                )[1]
            ))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


# --------------------------------------------------------------------------

import collections

Object = collections.namedtuple('Object', ['id', 'score', 'bbox', 'area', 'center', 'angles'])


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()


class Center(collections.namedtuple('Center', ['xcenter', 'ycenter'])):
    """Center.
    Utility to parse object center
    """
    __slots__ = ()


class Angles(collections.namedtuple('Angles', ['tx', 'ty'])):
    """Center.
    Utility to parse object center
    """
    __slots__ = ()

# --------------------------------------------------------------------------

# Labeling
import re


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


# --------------------------------------------------------------------------

# Making Interpreter
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(path, edgetpu):
    print(path, edgetpu)
    if (edgetpu == '0'):
        interpreter = tf.lite.Interpreter(model_path=path)
    else:
        path, *device = path.split('@')
        interpreter = tf.lite.Interpreter(model_path=path, experimental_delegates=[
            tf.lite.Interpreter.load_delegate(EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})])

    print('Loading Model: {} '.format(path))

    return interpreter


# --------------------------------------------------------------------------

def input_image_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    image = image.resize((input_image_size(interpreter)[0:2]), resample=Image.NEAREST)
    # input_tensor(interpreter)[:, :] = image

    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    # print(output_details)
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def invoke_interpreter(interpreter):
    global inference_time_ms

    t1 = time.time()
    interpreter.invoke()
    inference_time_ms = (time.time() - t1) * 1000
    print("****Inference time = ", inference_time_ms)


# --------------------------------------------------------------------------

def overlay_text_detection(objs, labels, cv2_im, fps):
    height, width, channels = cv2_im.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)
        percent = int(100 * obj.score)

        if (percent >= 60):
            box_color, text_color, thickness = (0, 255, 0), (0, 0, 0), 2
        elif (percent < 60 and percent > 40):
            box_color, text_color, thickness = (0, 0, 255), (0, 0, 0), 2
        else:
            box_color, text_color, thickness = (255, 0, 0), (0, 0, 0), 1

        text3 = '{}%; {}'.format(percent, labels.get(obj.id, obj.id))
        print(text3)

        try:
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), box_color, thickness)
            cv2_im = cv2.rectangle(cv2_im, (x0, y1 - 10), (x1, y1 + 10), (255, 255, 255), -1)
            cv2_im = cv2.putText(cv2_im, text3, (x0, y1), font, 0.6, text_color, thickness)
        except:
            # log_error()
            pass

    global model, inference_time_ms
    str1 = "FPS: " + str(fps)
    cv2_im = cv2.putText(cv2_im, str1, (width - 180, height - 55), font, 0.7, (255, 0, 0), 2)

    str2 = "Inference: " + str(round(inference_time_ms, 1)) + " ms"
    cv2_im = cv2.putText(cv2_im, str2, (width - 240, height - 25), font, 0.7, (255, 0, 0), 2)

    cv2_im = cv2.rectangle(cv2_im, (0, height - 20), (width, height), (0, 0, 0), -1)
    cv2_im = cv2.putText(cv2_im, model, (10, height - 5), font, 0.6, (0, 255, 0), 2)

    return cv2_im


# --------------------------------------------------------------------------

def main():
    interpreter = make_interpreter(model_path, edgetpu)

    interpreter.allocate_tensors()

    labels = load_labels(
        'models\\custom\\frc2023element_labels.txt')  # TODO fix os.path.join() parsing that adds two slashes for windows

    fps = 1

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        cv2_im = frame
        # cv2_im = cv2.flip(cv2_im, 0) # vertical reflection
        # cv2_im = cv2.flip(cv2_im, 1) # horizontal reflection

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_im_rgb)

        results = detect_objects(interpreter, image)
        cv2_im = overlay_text_detection(results, labels, cv2_im, fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('Green Vision Synapse', cv2_im)

        elapsed_ms = (time.time() - start_time) * 1000
        fps = round(1000 / elapsed_ms, 1)
        print("--------fps: ", fps, "---------------")


if __name__ == '__main__':
    main()

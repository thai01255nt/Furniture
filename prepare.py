import json, os, glob
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd

FURNITURE_RAW = "D:\\furniture_data"
IMAGES_DIR = os.path.join(FURNITURE_RAW, "data")
ANNOTATIONS_DIR = os.path.join(FURNITURE_RAW, "annotations")

with open(os.path.join(FURNITURE_RAW, "label_name_to_label_id.json")) as json_file:
    label_name_to_label_id = json.load(json_file)

with open(os.path.join(FURNITURE_RAW, "label_id_to_label_index.json")) as json_file:
    label_id_to_label_index = json.load(json_file)


class DatasetImagenet():
    def __init__(self, images_dir, annotations_dir, label_id_to_label_index):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir

        self.label_id_to_label_index = label_id_to_label_index
        self.list_label_id = list(label_id_to_label_index.keys())

        self.current_index_label_id = 0
        self.current_list_filename = glob.glob(
            os.path.join(self.annotations_dir, self.list_label_id[self.current_index_label_id]) + "/*")
        self.current_index_filename = 0

    def reset(self):
        self.current_index_label_id = 0
        self.current_list_filename = glob.glob(
            os.path.join(self.annotations_dir, self.list_label_id[self.current_index_label_id]) + "/*")
        self.current_index_filename = 0

    def next(self):
        if self.current_index_filename >= len(self.current_list_filename) - 1:
            if self.current_index_label_id >= len(self.list_label_id) - 1:
                return False
            self.current_index_label_id += 1
            self.current_list_filename = glob.glob(
                os.path.join(self.annotations_dir, self.list_label_id[self.current_index_label_id]) + "/*")
            self.current_index_filename = -1
        self.current_index_filename += 1
        return True

    def get_path(self):
        annotation_path = self.current_list_filename[self.current_index_filename]
        filename = os.path.splitext(os.path.basename(annotation_path))[0]
        image_path = os.path.join(self.images_dir, self.list_label_id[self.current_index_label_id], filename + ".JPEG")
        return image_path, annotation_path


class UtilsImagenet():
    def load_data_to_yolov5_format(image_path, annotation_path):
        image = cv2.imread(image_path)
        annotation = ET.parse(annotation_path)

        root = annotation.getroot()
        objects = root.findall("object")
        xsize = int(root.find("size/width").text)
        ysize = int(root.find("size/height").text)

        annotation = []
        for obj in objects:
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            xcenter = float(xmax + xmin) / 2 / xsize
            ycenter = float(ymax + ymin) / 2 / ysize
            width = float(xmax - xmin) / xsize
            height = float(ymax - ymin) / ysize
            annotation.append([name, xcenter, ycenter, width, height])
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        return image, annotation


dataset_furniture_imagenet = DatasetImagenet(images_dir=IMAGES_DIR, annotations_dir=ANNOTATIONS_DIR,
                                             label_id_to_label_index=label_id_to_label_index)

OUTPUT_TRAIN_DIR = "D:\\furniture_yolov5\\original\\train"
OUTPUT_VAL_DIR = "D:\\furniture_yolov5\\original\\val"
dataset_furniture_imagenet.reset()
num_val = {}
num_data = 0
while True:
    # print(dataset_furniture_imagenet.current_index_label_id,dataset_furniture_imagenet.current_index_filename,len(dataset_furniture_imagenet.current_list_filename))
    image_path, annotation_path = dataset_furniture_imagenet.get_path()

    if not os.path.exists(image_path):
        print(image_path)
        if not dataset_furniture_imagenet.next():
            break
        continue
    image, annotation = UtilsImagenet.load_data_to_yolov5_format(image_path=image_path, annotation_path=annotation_path)
    filename = os.path.splitext(os.path.basename(annotation_path))[0]
    annotation_out = []
    for anno in annotation:
        label_id = anno[0]
        if label_id in dataset_furniture_imagenet.label_id_to_label_index.keys():
            anno[0] = dataset_furniture_imagenet.label_id_to_label_index[label_id]
            annotation_out.append(anno)
        if anno[0] not in num_val.keys():
            num_val[anno[0]] = 0

    if num_val[anno[0]] <= 45:
        image_output_path = os.path.join(OUTPUT_VAL_DIR, filename + ".jpg")
        annotation_output_path = os.path.join(OUTPUT_VAL_DIR, filename + ".txt")
        num_val[anno[0]] += 1
    else:
        image_output_path = os.path.join(OUTPUT_TRAIN_DIR, filename + ".jpg")
        annotation_output_path = os.path.join(OUTPUT_TRAIN_DIR, filename + ".txt")
    cv2.imwrite(image_output_path, image)
    if len(annotation_out) > 0:
        annotation_out = pd.DataFrame(annotation_out)
        annotation_out.to_csv(annotation_output_path, header=False, index=False, sep=" ")
    num_data += 1
    if num_data % 50 == 0:
        print(filename)
        print(num_data)
    if not dataset_furniture_imagenet.next():
        break

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import xml.etree.ElementTree as ET

def xml_to_yolo(labels, C, img_size):
    l = []
    for i in labels:
        x_center = (i[1] + i[3])/2
        y_center = (i[2] + i[4])/2
        width = i[3] - i[1]
        height = i[4] - i[2]
        x_center = x_center/img_size[1]
        y_center = y_center/img_size[0]
        width = width/img_size[1]
        height = height/img_size[0]
        l.append(f"{C[i[0]]} {x_center} {y_center} {width} {height}"+"\n")
    return l


def extract_data(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    l = []
    for item in root.findall('object'):
        name = item.find('name').text
        bndbox = item.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        l.append([name, xmin, ymin, xmax, ymax])
    return l



def main(C, folder):
    l = os.listdir(folder)
    for i in range(0,len(l),2):
        img_path = os.path.join(folder, l[i])
        img_path = os.path.join(img_path, l[i])
        xml_path = os.path.join(folder, l[i+1])
        xml_path = os.path.join(xml_path, l[i+1])
        l2 = os.listdir(img_path)
        l3 = os.listdir(xml_path)
        os.makedirs(os.path.join(folder, l[i+1][0:-4]+'_yolo')) # for labels
        for j in range(len(l2)):
            img = cv2.imread(os.path.join(img_path, l2[j]))
            img_size = img.shape
            xml = os.path.join(xml_path, l3[j])
            labels = extract_data(xml)
            yolo_labels = xml_to_yolo(labels, C, img_size)
            with open(os.path.join(folder, l[i+1][0:-4]+'_yolo', l3[j].split('.')[0]+'.txt'), 'w') as f:
                f.writelines(yolo_labels)


folder = r"C:\Users\Lenovo\Desktop\IIIT-13K"

C = {
    'table': 0,
    'figure': 1,
    'natural_image': 2,
    'logo': 3,
    'signature': 4,
    }

main(C, folder)
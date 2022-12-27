"""
Reference Code : https://www.kaggle.com/code/clemchris/rbcd-pytorch-monai-train-infer
"""

import numpy as np
import pandas as pd

import pydicom
import cv2
import os
import re

from tqdm import tqdm

RESIZE_TO = (512, 512)

ROOT = 'D:/[DB]/Kaggle/rsna-breast-cancer-detection'


def dicom_file_to_image(path):
    dicom = pydicom.read_file(path)
    path_split = path.split('/')

    patient_id = re.compile('.dcm').sub('', path_split[-2])
    image_id = re.compile('.dcm').sub('', path_split[-1])

    data = dicom.pixel_array
    data = (data - data.min()) / (data.max() - data.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1 - data

    data = np.array(data)
    data = cv2.resize(data, RESIZE_TO)
    data = (data * 255).astype(np.uint8)

    output = f'{ROOT}/DCM2PNG/train_images/{folder}/{patient_id}_{image_id}.png'

    cv2.imwrite(output, data)

    return patient_id, image_id


count = 0
error_list = []
folders = os.listdir(f'{ROOT}/train_images')
for folder in folders:
    count += 1
    print(f'Folder Count [{count}/{len(folders)}]')

    images = os.listdir(f'{ROOT}/train_images/{folder}')
    os.makedirs(f'{ROOT}/DCM2PNG/train_images/{folder}', exist_ok=True)
    for image in images:
        try:
            patient_id, image_id = dicom_file_to_image(f'{ROOT}/train_images/{folder}/{image}')
            print(f'now image --> [patient_id : {patient_id}, image_id : {image_id}]')

        except Exception as e:
            patient_id = folder
            image_id = image
            print(f'Error : {e}')
            error_list.append([patient_id, image_id, e])


error_df = pd.DataFrame(data=error_list, columns=['patient_id', 'image_id', 'error_type'])
error_df.to_csv(f'{ROOT}/DCM2PNG/train_images/error_log.csv', index=False)
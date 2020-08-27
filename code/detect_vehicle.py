# Python version: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:

import sys
import os
import pickle
import json
import shutil
import random
import numpy as np
import pandas as pd
import re
import cv2
from shapely.geometry import Polygon
import gc

sys.path.append('/home/ming/Desktop/Satellite/aadt')

from dota.AerialDetection.DOTA_devkit import SplitOnlyImage_multi_process, DOTA2COCO


def split_img(file_path):
    print('making directory...')
    img_path = '{}/image'.format(file_path)
    split_path = '{}/split'.format(file_path)
    json_path = '{}/file'.format(file_path)
    try:
        os.mkdir(split_path)
    except FileExistsError:
        pass
    try:
        os.mkdir(json_path)
    except FileExistsError:
        pass

    print('splitting images into 1024*1024...')
    split = SplitOnlyImage_multi_process.splitbase(
        img_path,
        split_path,
        gap=200,
        subsize=1024,
        num_process=32
    )
    split.splitdata(1)
    DOTA2COCO.DOTA2COCOTest(
        split_path,
        os.path.join(json_path, 'test.json')
    )


def detect_car(
        file_path,
        test_file='/home/ming/Desktop/Satellite/code/dota/AerialDetection/tools/test.py',
        config_file='/home/ming/Desktop/Satellite/code/dota/AerialDetection/configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py',
        model='/home/ming/Desktop/Satellite/code/dota/DOTA_pytorch/model/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth'
):
    annfile = '{}/file/test.json'.format(file_path)
    imgprefix = '{}/split'.format(file_path)
    output = '/{}/file/output.pkl'.format(file_path)

    os.system(
        'python {} {} {} '
        '--out {} '  # pkl file
        '--annfile {} '  # json file
        '--imgprefix {}'.format(
            test_file, config_file, model,
            output, annfile, imgprefix)  # image path
    )


def pkl2csv(file_path):
    pkl_path = '{}/file/output.pkl'.format(file_path)
    dota_file = '{}/file/test.json'.format(file_path)
    output_file = '{}/file/test.csv'.format(file_path)
    print('convert pkl to csv...')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    with open(dota_file, 'r') as f:
        name = json.load(f)

    data = np.transpose(data)
    sv = data[4]
    lv = data[5]

    df_summary = pd.DataFrame()
    for i in range(len(sv)):
        df = pd.DataFrame(sv[i], columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'dif'])
        df['class'] = 4
        df['img'] = name['images'][i]['file_name']
        df_summary = pd.concat([df_summary, df], ignore_index=True)

        df = pd.DataFrame(lv[i], columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'dif'])
        df['class'] = 5
        df['img'] = name['images'][i]['file_name']
        df_summary = pd.concat([df_summary, df], ignore_index=True)

    df_summary.to_csv(output_file, index=None)
    del data, df_summary


def visualize(file_path):
    result_file = os.path.join(file_path, 'file', 'test.csv')
    df = pd.read_csv(result_file)
    group = df.groupby(['img'])
    for img_name, df_img in group:
        print(img_name)
        df_img = df_img[df_img.columns[:-1]]
        df_img[df_img < 0] = 0
        img_path = os.path.join(file_path, 'split', img_name)
        img = cv2.imread(img_path)
        for index in df_img.index:
            # print(df_img.loc[index])
            cv2.polylines(
                img,
                [
                    np.array([[int(df_img.loc[index, 'x1']), int(df_img.loc[index, 'y1'])],
                              [int(df_img.loc[index, 'x2']), int(df_img.loc[index, 'y2'])],
                              [int(df_img.loc[index, 'x3']), int(df_img.loc[index, 'y3'])],
                              [int(df_img.loc[index, 'x4']), int(df_img.loc[index, 'y4'])]])
                ],
                True,
                (0, 255, 0)
            )
            cv2.putText(
                img,
                str(df_img.loc[index, 'class']),
                (int(df_img.loc[index, 'x1']), int(df_img.loc[index, 'y1'])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0)
            )
        cv2.imshow('', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def merge(
        file_path,
        show_img=False):
    img = '{}/image/image.png'.format(file_path)
    output_file = '{}/file/test.csv'.format(file_path)

    df_box = pd.read_csv(output_file)
    df_box['width'] = df_box.img.apply(lambda x: int(re.findall(r'\d+', x)[-2]))
    df_box['height'] = df_box.img.apply(lambda x: int(re.findall(r'\d+', x)[-1]))
    df_box['x1'] = df_box.x1 + df_box.width
    df_box['x2'] = df_box.x2 + df_box.width
    df_box['x3'] = df_box.x3 + df_box.width
    df_box['x4'] = df_box.x4 + df_box.width
    df_box['y1'] = df_box.y1 + df_box.height
    df_box['y2'] = df_box.y2 + df_box.height
    df_box['y3'] = df_box.y3 + df_box.height
    df_box['y4'] = df_box.y4 + df_box.height

    if df_box.empty:
        cols = df_box.columns.tolist() + ['Polygon', 'area']
        df_box = df_box.reindex(columns=cols)
        return df_box
    df_box['Polygon'] = df_box.apply(lambda x: Polygon([(x.x1, x.y1), (x.x2, x.y2), (x.x3, x.y3), (x.x4, x.y4)]),
                                     axis=1)
    df_box['area'] = df_box['Polygon'].apply(lambda x: x.area)
    df_box = df_box[df_box['area'] > 440]

    if show_img:
        img = cv2.imread(img)
        for index in df_box.index:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.polylines(
                img,
                [
                    np.array([[int(df_box.loc[index, 'x1']), int(df_box.loc[index, 'y1'])],
                              [int(df_box.loc[index, 'x2']), int(df_box.loc[index, 'y2'])],
                              [int(df_box.loc[index, 'x3']), int(df_box.loc[index, 'y3'])],
                              [int(df_box.loc[index, 'x4']), int(df_box.loc[index, 'y4'])]])
                ],
                True,
                color,
                thickness=2
            )
            cv2.putText(
                img,
                str(df_box.loc[index, 'class']),
                (int(df_box.loc[index, 'x1']), int(df_box.loc[index, 'y1'])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255)
            )
        width = img.shape[1]
        height = img.shape[0]
        n = max(width//1096, height//1096)
        img = cv2.resize(img, (width//n, height//n), interpolation=cv2.INTER_CUBIC)
        cv2.imshow(' ', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return df_box


def main(file_path, show_img=False):
    split_path = '{}/file/test.json'.format(file_path)
    output = '{}/file/output.pkl'.format(file_path)
    output_file = '{}/file/test.csv'.format(file_path)

    if not os.path.exists(output_file):
        if not os.path.exists(output):
            if not os.path.exists(split_path):
                split_img(file_path)
            detect_car(file_path)
        pkl2csv(file_path)
    return merge(file_path, show_img)


if __name__ == '__main__':
    file_path = '/home/ming/Desktop/Satellite/code/output/43.668581,-79.394941'
    # split_img(file_path)
    # detect_car(file_path)
    # pkl2csv(file_path)
    # merge(file_path, show_img=True)
    main(file_path, show_img=True)

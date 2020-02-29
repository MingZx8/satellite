# Python vision: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:

# ######## ######## ######## ######## ######## ######### ######## ########
import pickle
import numpy as np
import pandas as pd
import json
import cv2
import os
import matplotlib.pyplot as plt


# ######## ######## ######## ######## ######## ######### ######## ########
# check if the val set is correctly labelled

# img_path = '/media/ming/data/dota_myset/test/images'
# label_path = '/media/ming/data/dota_myset/test/labelTxt'
# output_path = '/media/ming/data/dota_myset/test/images_bbox'
def check_label(img_path, label_path, output_path):
    for img_name in os.listdir(img_path):
        print(img_name)

        img = cv2.imread(os.path.join(img_path, img_name))
        label_file = os.path.join(label_path, img_name[:-4] + '.txt')
        df = pd.read_csv(label_file, sep=" ", header=None, skiprows=2)
        for index in df.index:
            if df.loc[index, 8] not in ['small-vehicle', 'large-vehicle']:
                continue
            cv2.polylines(
                img,
                [np.array([
                    [int(df.loc[index, 0]), int(df.loc[index, 1])],
                    [int(df.loc[index, 2]), int(df.loc[index, 3])],
                    [int(df.loc[index, 4]), int(df.loc[index, 5])],
                    [int(df.loc[index, 6]), int(df.loc[index, 7])],
                ])],
                isClosed=True,
                color=(0, 255, 0)
            )
            cv2.putText(
                img,
                str(df.loc[index, 9]),
                (int(df.loc[index, 0]), int(df.loc[index, 1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0)
            )
        cv2.imwrite(
            os.path.join(output_path, img_name),
            img
        )


# ######## ######## ######## ######## ######## ######### ######## ########
# convert .pkl to .csv

# pkl_file = '/home/ming/Desktop/Satellite/code/dota/DOTA_pytorch/output.pkl'
# dota_file = '/media/ming/data/dota/test1024/DOTA_test1024.json'

def pkl2csv(pkl_file, dota_file, output_file):
    print('convert pkl to csv')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    with open(dota_file, 'r') as f:
        name = json.load(f)

    # print(len(name['images']))
    # print(len(data))
    # print(len(data[0]))
    # print(len(data[0][0]))
    # print(len(data[0][0][0]))

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


# ######## ######## ######## ######## ######## ######### ######## ########
# visualize the result

# path = '/media/ming/data/dota/test1024/images'
# df = pd.read_csv('/media/ming/data/dota/test1024/test.csv')

def visualize(path, result_file, output_path):
    df = pd.read_csv(result_file)
    group = df.groupby(['img'])
    for img_name, df_img in group:
        print(img_name)
        df_img = df_img[df_img.columns[:-1]]
        df_img[df_img < 0] = 0
        img_path = os.path.join(path, img_name)
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
            # cv2.rectangle(img, (int(df_img.loc[index, 'x1']), int(df_img.loc[index, 'y1'])),
            #               (int(df_img.loc[index, 'x3']), int(df_img.loc[index, 'y3'])), (0, 255, 0), 1)
        filename = os.path.join(output_path, img_name)
        cv2.imwrite(filename, img)


if __name__ == '__main__':
    # # check if the val set is correctly labelled
    # img_path = '/media/ming/data/dota_myset/test/images'
    # label_path = '/media/ming/data/dota_myset/test/labelTxt'
    # output_path = '/media/ming/data/dota_myset/test/images_bbox'
    # check_label(img_path, label_path, output_path)

    # # convert .pkl to .csv
    # pkl_file = '/media/ming/data/dota_myset/test/result/results.pkl'
    # dota_file = '/media/ming/data/dota_1024/test1024/DOTA_test1024.json'
    # output_file = '/media/ming/data/dota_myset/test/result/results.csv'
    # pkl2csv(pkl_file, dota_file, output_file)


    # visualize the result
    img_path = '/media/ming/data/dota_1024/test1024/images'
    result_file = '/media/ming/data/dota_myset/test/result/results.csv'
    output_path = '/media/ming/data/dota_myset/test/result/images'
    visualize(img_path, result_file, output_path)

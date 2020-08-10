# Python vision: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020 Feb 13
# @Date modified: 2020 August 10
# Description: download satellite image from google image API
# Guide: https://developers.google.com/maps/documentation/maps-static/dev-guide

import urllib, requests
import pandas as pd
import numpy as np
import os, io
from PIL import Image
import cv2
import math
from convert import get_scale, dist2point


key = 'AIzaSyDQh-8JKJJaWgheQIUC4nU3jI8m2xNzpkI'


def get_url(latitude, longitude, key,
            width=640, height=640,
            zoom=19,
            scale=2,
            ):
    # location
    center = 'center={},{}'.format(latitude, longitude)  # the center of the map
    zoom = 'zoom={}'.format(zoom)  # the zoom level of the map

    # map
    size = "size={}x{}".format(width, height)  # the rectangular dimensions of the map image, max: 640*640*2(scale)
    scale = "scale={}".format(scale)  # the number of pixels, options: 1, 2, 4
    format = 'format=png'  # GIF and PNG provide greater detail
    maptype = 'maptype=satellite'

    # key
    key = 'key={}'.format(key)

    url = 'https://maps.googleapis.com/maps/api/staticmap?{}&{}&{}&{}&{}&{}&{}'.format(center, zoom, size, scale,
                                                                                       format, maptype, key)
    return url


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
    return image


def download(latitude, longitude, width, height, zoom, scale, path_output, key=key):
    '''
    download google maps image and write to path

    :param latitude: float
    :param longitude: float
    :param width: int
            >=640
    :param height: int
            >=640
    :param zoom: int
        from 1 to 22
    :param scale: int
        1 or 2
    :param path_output: str
    :param logo: bool
    :return: np.array
        img
    '''
    global count
    # TODO: if width or height < 640
    # TODO: remove logo
    n_width = math.ceil(width / 640) // 2
    n_height = math.ceil(height / 640) // 2
    resolution = get_scale(latitude, zoom, scale)
    dlat, dlon = dist2point(resolution * 640 * scale, resolution * 640 * scale, latitude)

    ls_lat = [dlat * _ + latitude for _ in list(range(n_height, -n_height - 1, -1))]
    ls_lon = [dlon * _ + longitude for _ in list(range(-n_width, n_width + 1))]
    ls_coordinate = list(zip(ls_lat * len(ls_lon), sorted(ls_lon * len(ls_lat))))

    print('{}, {}'.format(latitude, longitude))
    print('downloading...')
    ls_img = []
    for coord in ls_coordinate:
        url = get_url(coord[0], coord[1], key, 640, 640, zoom, scale)
        ls_img.append(url_to_image(url))
        count += 1

    print('concatenating...')
    ls_img_ver = []
    for i in range(len(ls_lon)):
        ls_img_ver.append(cv2.vconcat(ls_img[i * (len(ls_lat)):(i + 1) * (len(ls_lat))]))
    img = cv2.hconcat(ls_img_ver)

    height_head = (640 * (n_height * 2 + 1) - height) // 2 * scale
    width_head = (640 * (n_width * 2 + 1) - width) // 2 * scale
    img = img[height_head: height_head + height * scale, width_head: width_head + width * scale]
    cv2.imwrite(
        path_output,
        img
    )
    return img


if __name__ == '__main__':
    width = 2048
    height = 2048
    zoom = 19
    scale = 2

    neg_file = '/home/ming/Desktop/Satellite/data/ids_pred_2016_neg.txt'
    pos_file = '/home/ming/Desktop/Satellite/data/ids_pred_2016_pos.txt'
    coord_file = '/home/ming/Desktop/Satellite/data/coordinates.csv'
    path = '/media/ming/data/google_map_imgs/pred'
    df_coord = pd.read_csv(coord_file)

    neg_ls = []
    with open(neg_file, 'r') as f:
        for line in f.readlines():
            neg_ls.append(int(line))

    pos_ls = []
    with open(pos_file, 'r') as f:
        for line in f.readlines():
            pos_ls.append(int(line))

    point_ls = list(set(neg_ls + pos_ls))

    df_coord['neg'] = df_coord['centreline'].isin(neg_ls)
    df_coord['pos'] = df_coord['centreline'].isin(pos_ls)
    df_coord = df_coord[df_coord['neg'] | df_coord['pos']]

    df_coord.sort_values('centreline', inplace=True)
    for index in df_coord.index[:3600]:
        lat, lon = df_coord.loc[index, 'lat'], df_coord.loc[index, 'lon']
        centreline = df_coord.loc[index, 'centreline']
        output = os.path.join(path, str(centreline))
        print('centreline: ', centreline)
        if not os.path.exists(output):
            os.mkdir(output)
        if not os.path.exists(output + '/image.png'):
            download(
                lat, lon, width, height, zoom, scale, output + '/image.png', key=key
            )

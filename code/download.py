# Python vision: Python3.5
# @Author: MingZZZZZZZZ
# @Date created: 2020 Feb 13
# @Date modified: 2020 Feb 13
# Description:
# Api-key: AIzaSyC1tQUXzpoV0Wuj7E5ukEDfnLDDibbzxrg
# Guide: https://developers.google.com/maps/documentation/maps-static/dev-guide

import urllib, requests
import pandas as pd
import numpy as np
import os, io
from PIL import Image
import cv2
import math
from convert import get_scale, dist2point

key = 'AIzaSyC1tQUXzpoV0Wuj7E5ukEDfnLDDibbzxrg'


def get_url(latitude, longitude,
            width=640, height=640,
            zoom=19,
            scale=2,
            key=key
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

    return 'https://maps.googleapis.com/maps/api/staticmap?{}&{}&{}&{}&{}&{}&{}'.format(center, zoom, size, scale,
                                                                                        format, maptype, key)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
    return image


def download(latitude, longitude, width, height, zoom, scale, path_output='/media/ming/data/google_map_imgs'):
    if width <= 640 and height <= 640:
        url = get_url(latitude, longitude, width, height, zoom, scale, key)
        urllib.request.urlretrieve(url, '{}/{}*{}*{}*{}/{},{}.png'.format(
            path_output, width, height, scale, zoom, latitude, longitude))
        return
    n_width = math.ceil(width / 640) // 2
    n_height = math.ceil(height / 640) // 2
    resolution = get_scale(latitude, zoom, scale)
    dlat, dlon = dist2point(resolution * 640, resolution * 640, latitude)

    ls_lat = [dlat * _ + latitude for _ in list(range(n_height, -n_height - 1, -1))]
    ls_lon = [dlon * _ + longitude for _ in list(range(-n_width, n_width + 1))]
    ls_coordinate = list(zip(ls_lat * len(ls_lon), sorted(ls_lon * len(ls_lat))))

    print('{}, {}'.format(latitude, longitude))
    print('downloading...')
    ls_img = []
    for coord in ls_coordinate:
        url = get_url(coord[0], coord[1], 640, 640, zoom, scale, key)
        ls_img.append(url_to_image(url))

    print('concatenating...')
    ls_img_ver = []
    for i in range(len(ls_lon)):
        ls_img_ver.append(cv2.vconcat(ls_img[i * (len(ls_lat)):(i + 1) * (len(ls_lat))]))
    img = cv2.hconcat(ls_img_ver)

    height_head = (640 * (n_height * 2 + 1) - height) // 2
    width_head = (640 * (n_width * 2 + 1) - width) // 2
    img = img[height_head: height_head + height, width_head: width_head + width]
    if not os.path.exists('{}/{}*{}*{}*{}'.format(path_output, width, height, scale, zoom)):
        os.mkdir('{}/{}*{}*{}*{}'.format(path_output, width, height, scale, zoom))
    cv2.imwrite(
        '{}/{}*{}*{}*{}/{},{}.png'.format(
            path_output, width, height, scale, zoom, latitude, longitude),
        img
    )


if __name__ == '__main__':
    width = 1930
    height = 650
    zoom = 19
    scale = 1
    latitude = 43.74006
    longitude = -79.334

    # print(get_url(latitude, longitude, width, height, zoom, scale, key))
    download(latitude, longitude, width, height, zoom, scale)

    # path = '/home/ming/Desktop/Satellite/data/AADT example.csv'
    # aadt = pd.read_csv(path)
    #
    # dst_path = '/home/ming/Desktop/Satellite/data/imgs_google_maps/{}*{}*{}*{}'.format(width, height, scale, zoom)
    #
    # if not os.path.exists(dst_path):
    #     os.mkdir(dst_path)
    #
    # for index in aadt.index:
    #     latitude = round(aadt.loc[index, 'y.1'], 5)
    #     longitude = round(aadt.loc[index, 'x.1'], 4)
    #
    #     url = "https://maps.googleapis.com/maps/api/staticmap?{}".format(
    #         set_params(latitude=latitude, longitude=longitude,
    #                    width=width, height=height,
    #                    zoom=zoom,
    #                    scale=scale
    #                    ))
    #     print(url)
    #
    #     urllib.request.urlretrieve(url,
    #                                '{}/{},{}.png'.format(dst_path, latitude, longitude))

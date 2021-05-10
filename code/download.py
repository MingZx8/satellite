# Python version: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020 Feb 13
# @Date modified: 2020 August 10
# Description: download satellite image from google image API
# Guide: https://developers.google.com/maps/documentation/maps-static/dev-guide

import math
import os
import urllib.request

import cv2
import numpy as np

from convert import get_scale, dist2point

API_KEY = ''


def get_url(latitude, longitude,
            key=API_KEY,
            width=640, height=640, zoom=19, scale=2):
    """
    Setting 
    :param latitude: float
    :param longitude: float
    :param key: str, api key of google static map
    :param width: int
    :param height: int
    :param zoom: int, 1-22
    :param scale: int, 1 or 2
    :return: str, url for downloading the image
    """
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
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


def download(latitude, longitude,
             width, height,
             zoom, scale,
             output_path,
             key=API_KEY,
             centreline_label=''):
    """
    Download google maps satellite image and write to the path

    :param latitude: float
    :param longitude: float
    :param width: int
    :param height: int
    :param zoom: int, from 1 to 22
    :param scale: int, 1 or 2
    :param output_path: str
    :param key: str, optional, google map api key
    :param centreline_label: str, optional, by default, the file path is set as 'lat, lon', it will be named as given
    :return: np.array, image
    """
    # TODO: if width or height < 640
    # TODO: remove logo
    # create a folder, name it with the coordinate by default
    img_path = centreline_label if centreline_label else '{},{}'.format(latitude, longitude)
    try:
        os.mkdir(os.path.join(output_path, img_path))
    except FileExistsError:
        pass
    # create a image sub folder
    try:
        os.mkdir(os.path.join(output_path, img_path, 'image'))
    except FileExistsError:
        pass
    # name the image
    output_path = os.path.join(output_path, img_path, 'image', 'image.png')
    n_width = math.ceil(width / 640) // 2
    n_height = math.ceil(height / 640) // 2
    resolution = get_scale(latitude, zoom, scale)
    dlat, dlon = dist2point(resolution * 640 * scale, resolution * 640 * scale, latitude)

    ls_lat = [dlat * _ + latitude for _ in list(range(n_height, -n_height - 1, -1))]
    ls_lon = [dlon * _ + longitude for _ in list(range(-n_width, n_width + 1))]
    ls_coordinate = list(zip(ls_lat * len(ls_lon), sorted(ls_lon * len(ls_lat))))

    print('coordinate: {}, {}'.format(latitude, longitude))
    print('downloading...')
    ls_img = []
    for coord in ls_coordinate:
        url = get_url(coord[0], coord[1], key, 640, 640, zoom, scale)
        ls_img.append(url_to_image(url))

    print('concatenating...')
    ls_img_ver = []
    for i in range(len(ls_lon)):
        ls_img_ver.append(cv2.vconcat(ls_img[i * (len(ls_lat)):(i + 1) * (len(ls_lat))]))
    img = cv2.hconcat(ls_img_ver)

    height_head = (640 * (n_height * 2 + 1) - height) // 2 * scale
    width_head = (640 * (n_width * 2 + 1) - width) // 2 * scale
    img = img[height_head: height_head + height * scale, width_head: width_head + width * scale]
    cv2.imwrite(
        output_path,
        img
    )
    print('done.')
    return img


if __name__ == '__main__':
    # example
    download(43.659435, -79.354539, 2048, 1024, 19, 2, output_path='../output', centreline_label='eg')

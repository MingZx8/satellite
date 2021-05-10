# Python version: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2021
# Description:


import math


def get_scale(latitude, zoom=19, scale=2):
    """
    Given the latitude, zoom level and scale, return the scale in meter per pixel.

    :param latitude: float
    :param zoom: int
        from 1 to 22
    :param scale: int
        1 or 2
    :return: float
        meter per pixel
    """
    return 156543.03392 * math.cos(latitude * math.pi / 180) / math.pow(2, zoom) / scale


def point2dist(lat1, lon1, lat2, lon2):
    """
    Distance between 2 locations

    :param lat1: float
    :param lon1: float
    :param lat2: float
    :param lon2: float
    :return: tuple, (latitude distance, longitude distance)
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    def get_dist(dlat, lat1, lat2, dlon):
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6373.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1000

    return get_dist(lat2 - lat1, lat1, lat2, 0), get_dist(0, lat1, lat1, lon2 - lon1)  # lat, lon


def dist2point(dlat_m, dlon_m, lat):
    """
    Given distance in meter and latitude, return distance in coordinate

    :param dlat_m: float, change in meter for latitude
    :param dlon_m: float, change in meter for longitude
    :param lat: float
    :return: tuple, (change in latitude, change in longitude)
    """
    x = math.tan(dlat_m / 6373 / 2 / 1000) ** 2
    y = math.tan(dlon_m / 6373 / 2 / 1000) ** 2
    a = x / (x + 1)
    b = y / (y + 1)
    dlat = math.degrees(math.asin(math.sqrt(a)) * 2)
    dlon = math.degrees(math.asin(math.sqrt(b) / math.cos(math.radians(lat))) * 2)
    return dlat, dlon


def point2pixel(lat1, lon1, lat2, lon2, res):
    """
    given two points of coordinate and resolution, return the distance in pixel
    :param lat1: float, latitude of the first point
    :param lon1: float, longitude of the second point
    :param lat2: float, latitude of the first point
    :param lon2: float, longitude of the second point
    :param res: float, the scale of the image, can be calculated using function 'get_scale'
    :return: tuple, (lat, lon) in pixel
    """
    dist = point2dist(lat1, lon1, lat2, lon2)
    lat_dist = dist[0]
    lon_dist = dist[1]

    lat_pixel = lat_dist / res
    lon_pixel = lon_dist / res
    if lat1 < lat2:
        lat_pixel = - lat_pixel
    if lon1 > lon2:
        lon_pixel = - lon_pixel
    return lat_pixel, lon_pixel


def dist_in_image(pt1, pt2, img_height, img_width):
    """
    distance in pixel
    :param pt1: tuple, coordinates in the image in pixel
    :param pt2: tuple, coordinates in the image in pixel
    :param img_height: float, height of image in pixel
    :param img_width: float, width of image in pixel
    :return: float, distance of 2 points in pixel
    """
    lon_min = min(pt1[0], pt2[0])
    lon_max = max(pt1[0], pt2[0])
    lat_min = min(pt1[1], pt2[1])
    lat_max = max(pt1[1], pt2[1])

    def get_length(l_min, l_max, limit):
        if l_min < 0 <= l_max:
            length = l_max if l_max < limit else limit
        elif l_min < limit <= l_max:
            length = (limit - l_min) if l_min > 0 else limit
        else:
            length = 0 if (l_max < 0 or l_min > limit) else (l_max - l_min)
        return length

    return (get_length(lat_min, lat_max, img_height) ** 2 + get_length(lon_min, lon_max, img_width) ** 2) ** 0.5


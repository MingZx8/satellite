# Python vision: Python3.5
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from shapely.geometry import Polygon
import cv2
import os

path_img = '/home/ming/Desktop/Satellite/data/imgs_google_maps/512*512*2*19/result_improve/'
path_result = '/home/ming/Desktop/Satellite/data/imgs_google_maps/512*512*2*19/result_sf/'

# # ##############################assign geo info to AADT example################################################
# # ontario shapefile (time consuming)
# path_on = '/home/ming/Desktop/Satellite/data/ONrte/ONrte.shp'
# gdf_on = gpd.read_file(path_on)
#
# # aadt example location
# path_ex = '/home/ming/Desktop/Satellite/data/AADT example.csv'
# df_ex = pd.read_csv(path_ex).rename(columns={'y.1': 'Latitude', 'x.1': 'Longitude'})
# gdf_ex = gpd.GeoDataFrame(
#     df_ex,
#     geometry=gpd.points_from_xy(df_ex.Longitude, df_ex.Latitude)
# )
#
#
# def get_geo_index(point, shapefile=gdf_on):
#     return shapefile.geometry.distance(point).idxmin()
#
#
# gdf_ex['geo_index'] = gdf_ex.geometry.apply(get_geo_index)
# df_sum = pd.merge(df_ex.drop('geometry', axis=1), gdf_on.reset_index().rename(columns={'index': 'geo_index'}),
#                   on='geo_index', how='left')
# df_sum = gpd.GeoDataFrame(df_sum)
# df_sum.to_file('/home/ming/Desktop/Satellite/data/AADT_ex_geo.shp')

###########################################################################################################
# aadt example location with geo info
path_ex = '/home/ming/Desktop/Satellite/data/AADT_ex_geo.shp'
gdf_ex = gpd.read_file(path_ex)

gdf_point = gpd.GeoDataFrame(gdf_ex.drop('geometry', axis=1),
                             geometry=gpd.points_from_xy(gdf_ex.Longitude, gdf_ex.Latitude))


def point2dist(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    def get_dist(dlat, lat1, lat2, dlon):
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6373.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1000

    return get_dist(lat2 - lat1, lat1, lat2, 0), get_dist(0, lat1, lat1, lon2 - lon1)  # lat, lon


def get_scale(latitude, zoom=19, scale=2):
    return 156543.03392 * math.cos(latitude * math.pi / 180) / math.pow(2, zoom) / scale


def point2pixel(lat1, lon1, lat2, lon2, res):
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
    # longitude
    lon_min = min(pt1[0], pt2[0])
    lon_max = max(pt1[0], pt2[0])
    lat_min = min(pt1[1], pt2[1])
    lat_max = max(pt1[1], pt2[1])
    if lon_min < 0 and lon_max >= 0:
        len_lon = lon_max - 0 if lon_max < img_width else img_width
    elif lon_min < img_width and lon_max >= img_width:
        len_lon = img_width - lon_min if lon_min > 0 else img_width
    else:
        len_lon = 0 if lon_max < 0 or lon_min > img_width else lon_max - lon_min
    # latitude
    if lat_min < 0 and lat_max >= 0:
        len_lat = lat_max - 0 if lat_max < img_height else img_height
    elif lat_min < img_height and lat_max >= img_height:
        len_lat = img_height - lat_min if lat_min > 0 else img_height
    else:
        len_lat = 0 if lat_max < 0 or lat_min > img_height else lat_max - lat_min
    print(pt1, pt2, len_lat, len_lon)
    return (len_lon ** 2 + len_lat ** 2) ** 0.5


gdf_ex['m_per_pixel'] = gdf_ex['Latitude'].apply(get_scale)

j = 0
for index in gdf_ex.index:
    # the location coordinate
    test = gdf_ex.loc[index:index]
    test_point = gdf_point.loc[index:index]
    ax = test.plot()
    test_point.plot(ax=ax, color='orange')

    lon0 = test.Longitude.loc[index]
    lat0 = test.Latitude.loc[index]

    # draw roads to images based on shapefile data
    img_name = [_ for _ in os.listdir(path_img) if str(lat0)[:7] in _ and str(lon0)[:7] in _][0]
    img = cv2.imread(os.path.join(path_img + img_name))
    print(img_name)
    img_width = img.shape[0]
    img_height = img.shape[1]

    # validate the distance of lines with shapefile data
    dist = 0
    lines = []
    length_i = len(test.geometry.loc[index].xy[0])
    for i in range(length_i):
        lat1 = test.geometry.loc[index].xy[1][i]
        lon1 = test.geometry.loc[index].xy[0][i]
        if i < length_i - 1:
            lat2 = test.geometry.loc[index].xy[1][i + 1]
            lon2 = test.geometry.loc[index].xy[0][i + 1]

            tmp = point2dist(lat1, lon1, lat2, lon2)
            dist += math.sqrt(tmp[0] ** 2 + tmp[1] ** 2)

        # shapefile to pixel
        pixel = point2pixel(lat0, lon0, lat1, lon1, test.loc[index].m_per_pixel)
        loc = (pixel[1] + img_width / 2, pixel[0] + img_height / 2)
        lines.append(loc)

    plt.title(
        '{}_{}_{}'.format(gdf_ex.loc[index, 'STREETNAME'], gdf_ex.loc[index, 'SPD_KM'],
                          gdf_ex.loc[index, 'RDLEN_M_E']))
    # plt.show()
    plt.savefig(os.path.join(path_result, 'plt_' + img_name))
    plt.close()

    p_img = cv2.imread(os.path.join(path_result, 'plt_' + img_name))

    dist = 0
    for j in range(len(lines) - 1):
        # draw lines in images
        line1 = lines[j]
        line2 = lines[j + 1]
        cv2.line(
            img,
            tuple(map(int, line1)),
            tuple(map(int, line2)),
            (0, 0, 255),
            thickness=4
        )
        dist += dist_in_image(
            line1, line2,
            img_height,
            img_width
        ) * test.loc[index].m_per_pixel

    cv2.putText(
        img,
        '{} m'.format(round(dist, 1)),
        (100, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        thickness=2
    )

    # cv2.imshow('win', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # break

    cv2.imwrite(
        os.path.join(path_result, img_name),
        img
    )


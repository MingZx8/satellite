# Python version: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import cv2
import random
from shapely import wkt
from shapely.geometry import Point, Polygon, LineString

import road_mask
import detect_vehicle
import warnings
from convert import get_scale, point2pixel
from download import download

warnings.filterwarnings("ignore")


def main(
        latitude,
        longitude,
        width,
        height,
        zoom,
        scale,
        output_path,
        centreline_label='',
        show_count=False,
        geo_file='/media/ming/data/GeospatialData/ONrte/ONrte.shp'
):
    '''
    :param latitude: float
    :param longitude: float
    :param width: int
    :param height: int
    :param zoom: int,
        2~22, 19 is the best for car detection
    :param scale: int,
        1 or 2, 2 is the best for car detection
    :param output_path: str
        where the files are located
    :param centreline_label: str
        the label name of the road
    :param show_count: boolean
    :param geo_file: str
        where the geospatial data is located
    :return: None
    '''
    file_name = centreline_label if centreline_label else '{},{}'.format(latitude, longitude)
    file_path = os.path.join(output_path, file_name)
    img_file = '{}/image/image.png'.format(file_path)
    geo_selected_file = '{}/geo_val.csv'.format(file_path)
    count_img_file = '{}/count.png'.format(file_path)
    count_file = '{}/count.csv'.format(file_path)
    count_station_file = '{}/count_station.csv'.format(file_path)

    # ######## ######## ######## ######## ######## ######### ######## ########
    # download images
    if not os.path.exists(img_file):
        img = download(latitude,
                       longitude,
                       width,
                       height,
                       zoom,
                       scale,
                       output_path,
                       centreline_label=centreline_label
                       )
    else:
        img = cv2.imread(img_file)

    # image info
    img_width = img.shape[1]
    img_height = img.shape[0]
    n = max(img_width // 1096, img_height // 1096)
    resize_size = (img_width // n, img_height // n)
    resolution = get_scale(latitude)
    centroid = Point((int(img_width / 2), int(img_height / 2)))

    # ######## ######## ######## ######## ######## ######### ######## ########
    # load vehicle detection boxes
    df_box = detect_vehicle.main(file_path)

    #
    # generate mask
    if os.path.exists(geo_selected_file):
        df_mask = pd.read_csv(geo_selected_file, index_col=0)
        df_mask.dropna(subset=['mask'], inplace=True)
        df_mask['mask'] = df_mask['mask'].apply(wkt.loads)
        df_mask.geometry = df_mask.geometry.apply(wkt.loads)
    else:
        df_mask = road_mask.main(file_path,
                                 latitude,
                                 longitude,
                                 geo_file=geo_file
                                 )
        if isinstance(df_mask, type(None)):
            print('no geoinfo...')
            return
        df_mask.dropna(subset=['mask'], inplace=True)

    # initialize variables
    df_count = pd.DataFrame(
        columns=['STREET', 'DIRECTION', 'SPD', 'TRVLTIM', 'LENGTH', 'WIDTH', 'COUNT'])
    img_overlay = img.copy()

    # centerline
    center_line_dict = {}
    for line_index in df_mask.index:
        geo_info = df_mask.loc[line_index]
        center_line_geo = geo_info.geometry.xy
        center_line_geo = list(zip(center_line_geo[0], center_line_geo[1]))
        center_line = list(map(lambda x: point2pixel(latitude, longitude, x[1], x[0], resolution), center_line_geo))
        center_line = list(map(lambda x: (x[1] + img_width / 2, x[0] + img_height / 2), center_line))
        center_line_dict[line_index] = center_line

    point_geo = Point(longitude, latitude)

    # # show centerline
    # for j in center_line_dict:
    #     center_line = center_line_dict[j]
    #     for i in range(len(center_line[:-1])):
    #         cv2.line(
    #             img_overlay,
    #             tuple(map(int, center_line[i])),
    #             tuple(map(int, center_line[i + 1])),
    #             (0, 0, 127),
    #             thickness=5
    #         )
    # img_overlay = cv2.resize(img_overlay, resize_size, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('', img_overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # nearest to the station
    gdf = gpd.GeoDataFrame(df_mask, geometry='geometry')
    nearest_index = gdf.geometry.distance(point_geo).idxmin()
    nearest_geo = gdf.loc[nearest_index]
    # print('nearest street: ', nearest_geo.STREETNAME)

    # # 挑出最近的街
    df_street = gdf
    oneway = df_street[df_street['ONEWAY'] == 1]
    twoway = df_street.drop(oneway.index)
    oneway_ls = oneway.index.tolist()

    street_ls = []
    if oneway_ls:
        street_ls = [[oneway_ls[0]]]
        oneway_ls.pop(0)
    while oneway_ls:
        for i in oneway_ls:
            candidate = gdf.loc[i]
            chead = candidate.FROMNODE
            ctail = candidate.TONODE
            flag = 0
            for ls in street_ls:
                head = gdf.loc[ls[0], 'FROMNODE']
                tail = gdf.loc[ls[-1], 'TONODE']
                if chead == tail:
                    ls.append(i)
                    oneway_ls.remove(i)
                    flag = 1
                    break
                elif ctail == head:
                    ls.insert(0, i)
                    oneway_ls.remove(i)
                    flag = 1
                    break
            if flag:
                break
        if not flag:
            street_ls.append([oneway_ls[0]])
            oneway_ls.pop(0)
    side_all = {0: [], 1: []}
    paralle_flag = 0
    for ls in street_ls:
        # 是否是一个环
        head = gdf.loc[ls[0], 'FROMNODE']
        tail = gdf.loc[ls[-1], 'TONODE']
        circle = 1 if head == tail else 0

        i = 0
        flag = 0
        if circle:
            for node in ls:
                head_node = gdf.loc[node, 'FROMNODE']
                for index in twoway.index:
                    if df_street.loc[index, 'TONODE'] == head_node:
                        ls = ls[i:] + ls[:i]
                        flag = 1
                        break
                i += 1
                if flag:
                    break
        # print('street_circle:', ls)

        side = {0: [], 1: []}
        flag = 0
        for node in ls:
            head_node = gdf.loc[node, 'FROMNODE']
            for index in twoway.index:
                if df_street.loc[index, 'FROMNODE'] == head_node:
                    if not circle:
                        side[1 - flag], side[flag] = side[flag], side[1 - flag]
                    else:
                        flag = 0 if flag else 1
                if df_street.loc[index, 'TONODE'] == head_node:
                    flag = 0 if flag else 1
            side[flag].append(node)

        if circle and len(side[0]) != len(side[1]):
            short = 0 if len(side[0]) < len(side[1]) else 1
            long = 1 - short
            ave = int((len(side[0]) + len(side[1])) / 2)
            side[short] += side[long][ave:]
            side[long] = side[long][:ave]
        if twoway.empty and paralle_flag == 1:
            side_all[1] += side[0]
            side_all[0] += side[1]
        else:
            side_all[0] += side[0]
            side_all[1] += side[1]
        paralle_flag += 1

    # print(side_all)

    pool = []
    box_on_road = {1: pd.DataFrame(), 0: pd.DataFrame()}

    mask_right_all = Polygon([(0, 0), (0, 0), (0, 0)])
    mask_left_all = Polygon([(0, 0), (0, 0), (0, 0)])

    def count_under_mask(mask):
        df_box[mask_index] = df_box['Polygon'].apply(lambda x: x.centroid.intersects(mask))
        df_box_selected = df_box[df_box[mask_index]]
        count = 0
        car_counted_indexes = []
        for box_index in df_box_selected.index:
            if box_index in pool:
                continue
            count += 1
            car_counted_indexes.append(box_index)
            pool.append(box_index)
            for other_box_index in df_box_selected.drop(box_index).index:
                if df_box_selected.loc[box_index, 'Polygon'].intersection(
                        df_box_selected.loc[other_box_index, 'Polygon']).area >= 200:
                    pool.append(other_box_index)
        return count, car_counted_indexes

    color_right = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color_left = (0, 0, 0)
    df_mask.loc[:, 'vehicle_1'] = None
    df_mask.loc[:, 'vehicle_0'] = None
    for mask_index in df_street.index:
        mask = df_mask.loc[mask_index, 'mask']
        if not mask:
            continue

        if mask_index in side_all[0] + side_all[1]:
            direction = 0 if mask_index in side_all[0] else 1
            color = color_right if direction else color_left
            if direction:
                mask_right_all = mask_right_all.union(mask)
            else:
                mask_left_all = mask_left_all.union(mask)
            vehicles = count_under_mask(mask)
            df_mask.loc[mask_index, 'vehicle_{}'.format(direction)] = vehicles[0]
            box_on_road[direction] = pd.concat([box_on_road[direction], df_box.loc[vehicles[1]]])

            mask_buffer = list(
                zip(map(int, mask.exterior.coords.xy[0]), map(int, mask.exterior.coords.xy[1])))
            cv2.drawContours(
                img_overlay,
                [np.array(mask_buffer)],
                -1,
                color,
                thickness=cv2.FILLED
            )
        else:
            mask_right = mask.intersection(
                LineString(center_line_dict[mask_index]).buffer(1000, cap_style=2, join_style=2, single_sided=True))
            mask_left = mask.intersection(
                LineString(center_line_dict[mask_index]).buffer(-1000, cap_style=2, join_style=2, single_sided=True))
            vehicles_right = count_under_mask(mask_right)
            vehicles_left = count_under_mask(mask_left)
            df_mask.loc[mask_index, 'vehicle_1'] = vehicles_right[0]
            df_mask.loc[mask_index, 'vehicle_0'] = vehicles_left[0]
            box_on_road[1] = pd.concat([box_on_road[1], df_box.loc[vehicles_right[1]]])
            box_on_road[0] = pd.concat([box_on_road[0], df_box.loc[vehicles_left[1]]])

            mask_right_all.union(mask_right)
            mask_left_all.union(mask_left)

            try:
                mask_left_buffer = list(
                    zip(map(int, mask_left.exterior.coords.xy[0]), map(int, mask_left.exterior.coords.xy[1])))
                cv2.drawContours(
                    img_overlay,
                    [np.array(mask_left_buffer)],
                    -1,
                    color_left,
                    thickness=cv2.FILLED
                )
            except AttributeError:
                pass
            try:
                mask_right_buffer = list(
                    zip(map(int, mask_right.exterior.coords.xy[0]), map(int, mask_right.exterior.coords.xy[1])))
                cv2.drawContours(
                    img_overlay,
                    [np.array(mask_right_buffer)],
                    -1,
                    color_right,
                    thickness=cv2.FILLED
                )
            except AttributeError:
                pass

    # draw vehicles
    for index in df_box.index:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(
            img_overlay,
            [np.array(
                [[int(df_box.loc[index, 'x1']), int(df_box.loc[index, 'y1'])],
                 [int(df_box.loc[index, 'x2']), int(df_box.loc[index, 'y2'])],
                 [int(df_box.loc[index, 'x3']), int(df_box.loc[index, 'y3'])],
                 [int(df_box.loc[index, 'x4']), int(df_box.loc[index, 'y4'])]]
            )],
            -1,
            color,
            thickness=cv2.FILLED
        )
    cv2.addWeighted(
        img_overlay, 0.5, img, 0.5, 0, img_overlay
    )
    img_mask = cv2.resize(img_overlay, resize_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(count_img_file, img_mask)

    if show_count:
        cv2.imshow(' ', img_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # generate table
    df_street = df_mask[df_mask['STREETNAME'] == nearest_geo.STREETNAME]
    street = nearest_geo.STREETNAME
    df_street = df_street[df_street['width'] != 0]
    direction = 0 if mask_right_all.distance(centroid) > mask_left_all.distance(centroid) else 1  # left is positive
    df_pos = df_street[~df_street['vehicle_{}'.format(direction)].isnull()]
    df_neg = df_street[~df_street['vehicle_{}'.format(1 - direction)].isnull()]
    pos_speed = df_pos['SPD_KM'].unique()
    neg_speed = df_neg['SPD_KM'].unique()
    pos_length = df_pos['length'].sum()
    neg_length = df_neg['length'].sum()
    pos_count = df_pos['vehicle_{}'.format(direction)].sum()
    neg_count = df_neg['vehicle_{}'.format(1 - direction)].sum()
    pos_trvltime = (df_pos['length'] / df_pos['SPD_KM'] / 1000).sum()
    neg_trvltime = (df_neg['length'] / df_neg['SPD_KM'] / 1000).sum()
    pos_width = pd.concat(
        [df_pos[df_pos['ONEWAY'] == 0]['width_{}'.format(direction)], df_pos[df_pos['ONEWAY'] == 1]['width']]).mean()
    neg_width = pd.concat([df_neg[df_neg['ONEWAY'] == 0]['width_{}'.format(1 - direction)],
                           df_neg[df_neg['ONEWAY'] == 1]['width']]).mean()
    # ['STREET', 'DIRECTION', 'SPD', 'TRVLTIM', 'LENGTH', 'WIDTH', 'COUNT']
    df_count.loc[0] = [street, 'p', pos_speed, pos_trvltime, pos_length, pos_width, pos_count]
    df_count.loc[1] = [street, 'n', neg_speed, neg_trvltime, neg_length, neg_width, neg_count]

    df_mask.to_csv(count_file)
    df_count.to_csv(count_station_file)

    box_columns = ['direction', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'centroid_x', 'centroid_y', 'class',
                   'area']

    for i in [0, 1]:
        try:
            box_on_road[i]['centroid_x'] = box_on_road[i]['Polygon'].apply(lambda x: x.centroid.x)
            box_on_road[i]['centroid_y'] = box_on_road[i]['Polygon'].apply(lambda x: x.centroid.y)
            box_on_road[i]['direction'] = 'p' if direction == i else 'n'
            box_on_road[i] = box_on_road[i][box_columns]
        except KeyError:
            pass

    box_target = pd.concat([box_on_road[0], box_on_road[1]])
    box_target.to_csv('{}/box.csv'.format(file_path))


if __name__ == '__main__':
    main(
        43.659435, -79.354539,
        2048,
        1024,
        19,
        2,
        '../output',
        centreline_label='eg'
    )

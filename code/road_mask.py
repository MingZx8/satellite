# Python version: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from shapely import wkt
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import difflib
import os
from sklearn.cluster import KMeans

from convert import point2pixel, get_scale, dist_in_image, dist2point
from generateLSD import generate

import random


def get_geo_index(point, gdf):
    return gdf.geometry.distance(point).idxmin()


def pt2line(pt1, pt2):
    return LineString(((pt1[0], pt1[1]), (pt2[0], pt2[1])))


def get_angle(k1, k2, pair=True):
    angle = math.atan((k1 - k2) / (1 + k1 * k2)) / math.pi * 360
    if pair:
        return 0 if abs(angle) > 3 else 1
    return 1 if abs(angle) < 45 else 0


def get_perpendicular_point(x0, y0, a, c):
    return (x0 + a * y0 - a * c) / (a ** 2 + 1), (a ** 2 * x0 + a * x0 + c) / (a ** 2 + 1)


def get_vertical_point(pt, a, c):
    x = (pt[0] + a * pt[1] - a * c) / (a ** 2 + 1)
    return x, a * x + c


def get_parallel_dist(pt0, pt1, pt2):
    try:
        a = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    except ZeroDivisionError:
        return abs(pt0[0] - pt1[0]), pt1[0], pt0[1]
    c = pt1[1] - pt1[0] * a
    dist = ((a * pt0[0] - pt0[1] + c) ** 2 / (a ** 2 + 1)) ** 0.5
    x, y = get_vertical_point(pt0, a, c)
    return dist, x, y


def get_overlap_ratio(df_segment, fix_step, target_id, candidate_id):
    target = df_segment.loc[target_id]
    candidate = df_segment.loc[candidate_id]
    a = candidate.slope
    c = candidate.intercept

    # sample points on the target line
    slope = df_segment.loc[target_id, 'slope']
    intercept = df_segment.loc[target_id, 'intercept']
    pt_ls = []
    left = right = 1
    pt_x_r = target.x1 + (fix_step ** 2 / (slope ** 2 + 1)) ** 0.5
    pt_x_l = target.x1 - (fix_step ** 2 / (slope ** 2 + 1)) ** 0.5
    if pt_x_r > max(target.x1, target.x2):
        right = 0
    if pt_x_l < min(target.x1, target.x2):
        left = 0
    pt_x = target.x1
    while left or right:
        pt_ls.append((pt_x, slope * pt_x + intercept))
        pt_x = pt_x + right * (fix_step ** 2 / (slope ** 2 + 1)) ** 0.5 - left * (
                fix_step ** 2 / (slope ** 2 + 1)) ** 0.5
        if pt_x > max(target.x1, target.x2):
            right = 0
        if pt_x < min(target.x1, target.x2):
            left = 0

    vertical_point_ls = [_ for _ in [get_vertical_point(i, a, c) for i in pt_ls]
                         if min(candidate.x1, candidate.x2) <= _[0] <= max(candidate.x1, candidate.x2)]
    try:
        overlap_ratio = len(vertical_point_ls) / len(pt_ls)  # overlap ratio
    except ZeroDivisionError:
        return 0
    return 0 if overlap_ratio < 0.2 else 1


def get_distance(df_segment, line1_id, line2_id, resolution):
    line1 = df_segment.loc[line1_id]
    line2 = df_segment.loc[line2_id]
    a = line2.slope
    c = line2.intercept
    pt1_x, pt1_y = line1.x1, line1.y1
    pt2_x = (pt1_x + a * pt1_y - a * c) / (a ** 2 + 1)
    pt2_y = a * pt2_x + c
    return ((pt1_y - pt2_y) ** 2 + (pt1_x - pt2_x) ** 2) ** 0.5 * resolution


def get_trans_dist(center_line, pt0, resolution):
    dict_line = {}
    tmp_dist = []
    for i in range(len(center_line[:-1])):
        pt10 = center_line[i]
        pt20 = center_line[i + 1]
        dist, x, y = get_parallel_dist(pt0, pt10, pt20)
        if min(pt10[0], pt20[0]) <= x <= max(pt10[0], pt20[0]) and \
                min(pt10[1], pt20[1]) <= y <= max(pt10[1], pt20[1]):
            tmp = dist * resolution
            tmp_dist.append(tmp)
            dict_line[i] = [x, y, tmp]
    x, y = [(x, y) for i, (x, y, dist) in dict_line.items() if dist == min(tmp_dist)][0]
    return x - pt0[0], y - pt0[1]


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def main(latitude, longitude,
         img_scale='1024*1024*2*19',
         show_selected_area=False,
         show_filtered_segment=False,
         show_paired_segment=False,
         show_mask=False,
         validate=False,
         centreline_label=None,
         path='/media/ming/data/google_map_imgs',
         geo_path='/media/ming/data/GeospecialData/ONrte',
         gdf_all=None
         ):
    resolution = get_scale(latitude)
    # file loading#################################################################################
    file_name = '{}'.format(centreline_label) if centreline_label else '{},{}'.format(latitude, longitude)
    file_path = '{}/{}/{}'.format(path, img_scale, file_name)
    img_name = '{}/image/image.png'.format(file_path)
    geo_name = '{}/ONrte.shp'.format(geo_path)
    geo_selected_name = '{}/{}'.format(geo_path, file_name)
    geo_line_name = '{}/line.csv'.format(file_path)
    line_name = '{}/lsd.txt'.format(file_path)
    geo_file = '{}/{}/{}/geo_val.csv'.format(path, img_scale, file_name) if validate else \
        '{}/{}/{}/geo.csv'.format(path, img_scale, file_name)

    default_polygon = Polygon([(0, 0), (0, 0), (0, 0)])

    # load image
    img = cv2.imread(img_name)
    img_mask = img.copy()
    img_width = img.shape[1]
    img_height = img.shape[0]

    # generate lsd file
    if not os.path.exists(line_name):
        print('generate line segment...')
        generate(latitude, longitude, img_scale, centreline_label=centreline_label, path=path)

    # build line segment dataframe
    df_segment = pd.read_csv(line_name, sep=' ', header=None).rename(
        columns=dict(enumerate(['x1', 'y1', 'x2', 'y2', 'width', 'p', '-log_nfa']))).drop(columns=7)
    df_segment['slope'] = (df_segment.y2 - df_segment.y1) / (df_segment.x2 - df_segment.x1)
    df_segment['intercept'] = df_segment.y1 - df_segment.slope * df_segment.x1
    df_segment['width'] = ((df_segment.y2 - df_segment.y1) ** 2 + (df_segment.x2 - df_segment.x1) ** 2) ** 0.5

    # split/read geospatial info
    print('loading geospatial file...')
    if os.path.exists(geo_selected_name):
        file_ls = os.listdir(geo_selected_name)
        for file in file_ls:
            if '.csv' in file:
                gdf = pd.read_csv(os.path.join(geo_selected_name, file))
                try:
                    gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
                except KeyError:
                    pass
                break
            if '.shp' in file:
                gdf = gpd.read_file(os.path.join(geo_selected_name, file))
                break
        if gdf.empty:
            return
        try:
            gdf.set_index('flag', inplace=True)
        except KeyError:
            gdf = gdf_all if gdf_all else gpd.read_file(geo_name)
            point = Point(longitude, latitude).buffer(0.01)
            gdf['selected'] = gdf.geometry.apply(point.intersects)
            gdf = gdf[gdf.selected]
            gdf['flag'] = gdf.index
            gdf.to_file(geo_selected_name)
    else:
        gdf = gpd.read_file(geo_name) if isinstance(gdf_all, type(None)) else gdf_all
        point = Point(longitude, latitude).buffer(0.01)
        gdf['selected'] = gdf.geometry.apply(point.intersects)
        gdf = gdf[gdf.selected]
        gdf['flag'] = gdf.index
        if gdf.empty:
            os.mkdir(geo_selected_name)
            gdf.to_csv('{}/{}'.format(geo_selected_name, file_name + '.csv'))
            return
        else:
            gdf.to_file(geo_selected_name)
    print('done')

    ############################################################################################
    # get 1. fixed distance
    # 2. depends on nearby shapefile line
    # display all lines in the image
    dlat_m = resolution * img_height / 2
    dlon_m = resolution * img_width / 2

    delta = dist2point(dlat_m, dlon_m, latitude)
    pt1 = (longitude - delta[1], latitude - delta[0])
    pt2 = (longitude + delta[1], latitude - delta[0])
    pt3 = (longitude - delta[1], latitude + delta[0])
    pt4 = (longitude + delta[1], latitude + delta[0])
    poly_img = Polygon([pt1, pt2, pt4, pt3])

    gdf['display'] = gdf.geometry.apply(poly_img.intersects)
    gdf_line = gdf[gdf['display']]
    gdf_line.fillna('None', inplace=True)
    gdf_line.to_csv(geo_line_name)
    if gdf_line.empty:
        return

    # if validate, choose the nearest line to the centre point
    if validate:
        nearest_index = get_geo_index(Point(longitude, latitude), gdf_line)
        gdf_line_selected = gdf_line[gdf_line['STREETNAME'] == gdf_line.loc[nearest_index, 'STREETNAME']]
        index_ls = gdf_line_selected.index.to_list()
        index_ls.remove(nearest_index)
        nearest_line_ls = [nearest_index]
        len_tmp = len(nearest_line_ls)
        head = gdf_line_selected.loc[nearest_index, 'FROMNODE']
        tail = gdf_line_selected.loc[nearest_index, 'TONODE']
        node_ls = [head, tail]
        while len(nearest_line_ls) == len_tmp:
            len_tmp += 1
            for i in index_ls:
                head = gdf_line_selected.loc[i, 'FROMNODE']
                tail = gdf_line_selected.loc[i, 'TONODE']
                if head in node_ls or tail in node_ls:
                    nearest_line_ls.append(i)
                    index_ls.remove(i)
                    node_ls.append(head)
                    node_ls.append(tail)
                    break
        gdf_line_selected = gdf_line_selected.loc[nearest_line_ls, :]
    else:
        gdf_line_selected = gdf_line

    # creating center lines for all gdf_line
    center_line_dict = {}
    for line_index in gdf_line.index:
        geo_info = gdf_line.loc[line_index]
        center_line_geo = geo_info.geometry.xy
        center_line_geo = list(zip(center_line_geo[0], center_line_geo[1]))
        center_line = list(map(lambda x: point2pixel(latitude, longitude, x[1], x[0], resolution), center_line_geo))
        center_line = list(map(lambda x: (x[1] + img_width / 2, x[0] + img_height / 2), center_line))
        center_line_dict[line_index] = center_line

    for line_index in gdf_line_selected.index:
        geo_info = gdf_line.loc[line_index]
        # print(line_index)
        print('creating detecting area...')
        # default road width setting
        lane_limit = 3 if (geo_info.CARTO < 4 and geo_info.SPD_KM < 70) else \
            2.2 if geo_info.CARTO == 5 else \
                1.5 if geo_info.CARTO == 6 else \
                    4 if geo_info.ONEWAY else 5
        offset = 3.5 * lane_limit / resolution
        # the center line
        center_line = center_line_dict[line_index]
        center_line_slope = []
        for i in range(len(center_line) - 1):
            try:
                center_line_slope.append(
                    (center_line[i][1] - center_line[i + 1][1]) / (center_line[i][0] - center_line[i + 1][0]))
            except ZeroDivisionError:
                pass

        # creating selected area
        # boundary including other lines
        boundary_area = 0
        for other_line_index in center_line_dict:
            if other_line_index == line_index:
                continue
            other_line = center_line_dict[other_line_index]
            other_line_buffer = LineString(other_line).buffer(3.5 / resolution, cap_style=2, join_style=2)
            if boundary_area == 0:
                boundary_area = other_line_buffer
            else:
                boundary_area = boundary_area.union(other_line_buffer)
        if boundary_area == 0:
            boundary_area = default_polygon

        def get_mask(selected_area, selected_area1, side='', signal=1):
            # draw selected area
            overlay = img.copy()
            if type(selected_area) == Polygon:
                try:
                    center_line_buffer = list(
                        zip(map(int, selected_area.exterior.coords.xy[0]),
                            map(int, selected_area.exterior.coords.xy[1])))
                    cv2.drawContours(
                        overlay,
                        [np.array(center_line_buffer)],
                        -1,
                        (130, 197, 82),
                        thickness=cv2.FILLED)
                except AttributeError:
                    pass
            else:
                new_selected_area = selected_area
                for poly in selected_area:
                    if not LineString(center_line).intersects(poly):
                        new_selected_area = new_selected_area.difference(poly)
                    else:
                        poly_buffer = list(
                            zip(map(int, poly.exterior.coords.xy[0]),
                                map(int, poly.exterior.coords.xy[1])))
                        cv2.drawContours(
                            overlay,
                            [np.array(poly_buffer)],
                            -1,
                            (130, 197, 82),
                            thickness=cv2.FILLED)
                selected_area = new_selected_area
            line_length = 0
            for i in range(len(center_line[:-1])):
                cv2.line(
                    overlay,
                    tuple(map(int, center_line[i])),
                    tuple(map(int, center_line[i + 1])),
                    (0, 0, 127),
                    thickness=5
                )
                line_length += dist_in_image(center_line[i], center_line[i + 1], img_height, img_width) * resolution
            cv2.addWeighted(
                overlay, 0.5, img, 0.5, 0, overlay
            )
            overlay = cv2.resize(overlay, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            if show_selected_area:
                cv2.imshow(' ', overlay)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # filter out line segment out of the selected area and its length should be larger than 3 meters
            df_segment['geometry'] = df_segment.apply(lambda x: pt2line((x.x1, x.y1), (x.x2, x.y2)), axis=1)
            df_segment['filtered'] = df_segment.geometry.apply(selected_area.intersects)
            df_segment['filtered2'] = df_segment['slope'].apply(
                lambda x: sum([get_angle(x, i, pair=False) for i in center_line_slope]) > 0)
            pair_name = '{}/lsd_pair_{}{}.csv'.format(file_path, line_index, side)
            pair_refine_name = '{}/lsd_refine_{}{}.csv'.format(file_path, line_index, side)
            mask_name = '{}/mask_{}{}.png'.format(file_path, line_index, side)

            # TODO: remove google map logo
            # minimum line segment setting
            df_segment_filter = df_segment[df_segment.filtered & (df_segment.width > (
                5 / resolution if geo_info.CARTO < 4 else
                3 / resolution if geo_info.CARTO in [5, 6] else
                4 / resolution))
                                           & df_segment.filtered2]

            if show_filtered_segment:
                img_copy = img.copy()
                for i in df_segment_filter.index:
                    cv2.line(img_copy,
                             tuple(map(int, (df_segment_filter.loc[i, 'x1'], df_segment_filter.loc[i, 'y1']))),
                             tuple(map(int, (df_segment_filter.loc[i, 'x2'], df_segment_filter.loc[i, 'y2']))),
                             (23, 117, 187),
                             thickness=3
                             )
                img_copy = cv2.resize(img_copy, (1024, 1024), interpolation=cv2.INTER_CUBIC)
                cv2.imshow(' ', img_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            ###############################################################################################
            print('pairing segments...')
            # pair by slope

            if os.path.exists(pair_name):
                df_segment_pair = pd.read_csv(pair_name, index_col=0)
            else:
                df_segment_pair = pd.DataFrame(columns=df_segment_filter.index)
                for i in df_segment_filter.index:
                    for j in df_segment_filter.index.drop(i):
                        df_segment_pair.loc[i, j] = get_angle(df_segment_filter.loc[i, 'slope'],
                                                              df_segment_filter.loc[j, 'slope'])
                df_segment_pair.to_csv(pair_name)

            # refining the pairs
            print('refining...')
            fix_step = df_segment_filter.width.mean() / 10
            if os.path.exists(pair_refine_name):
                df_segment_pair = pd.read_csv(pair_refine_name, index_col=0)
            else:
                for col in df_segment_pair.columns:
                    for row in df_segment_pair.index:
                        poly = Polygon(
                            [(df_segment.loc[int(row)].x1, df_segment.loc[int(row)].y1),
                             (df_segment.loc[int(row)].x2, df_segment.loc[int(row)].y2),
                             (df_segment.loc[int(col)].x1, df_segment.loc[int(col)].y1),
                             (df_segment.loc[int(col)].x2, df_segment.loc[int(col)].y2)]
                        )
                        df_segment_pair.loc[row, col] = get_overlap_ratio(df_segment, fix_step, int(row), int(col)) \
                                                        and get_overlap_ratio(df_segment, fix_step, int(col), int(row)) \
                                                        and df_segment_pair.loc[row, col]
                df_segment_pair.to_csv(pair_refine_name)

            # get distance for each pair
            dist_dict = {}
            for col in df_segment_pair.columns:
                for row in df_segment_pair.index:
                    if df_segment_pair.loc[row, col] == 1:
                        df_segment_pair.loc[row, col] = get_distance(df_segment, int(row), int(col), resolution)
                        if df_segment_pair.loc[row, col] <= 3 or df_segment_pair.loc[row, col] >= 3 * 7:
                            df_segment_pair.loc[row, col] = 0
                        else:
                            dist_dict[(row, col)] = df_segment_pair.loc[row, col]

            # show paired segments
            if show_paired_segment:
                pair_ls_all = []
                for col in df_segment_pair.columns:
                    for row in df_segment_pair.index:
                        try:
                            if df_segment_pair.loc[row, col] > 0 and df_segment_pair.loc[int(col), str(row)] > 0:
                                pair_ls_all.append(int(row))
                                pair_ls_all.append(int(col))
                        except KeyError:
                            if df_segment_pair.loc[row, col] > 0 and df_segment_pair.loc[col, row] > 0:
                                pair_ls_all.append(int(row))
                                pair_ls_all.append(int(col))

                img_copy = img.copy()
                for i in set(pair_ls_all):
                    cv2.line(img_copy,
                             tuple(map(int, (df_segment_filter.loc[i, 'x1'], df_segment_filter.loc[i, 'y1']))),
                             tuple(map(int, (df_segment_filter.loc[i, 'x2'], df_segment_filter.loc[i, 'y2']))),
                             (253, 200, 84),
                             thickness=3
                             )
                img_copy = cv2.resize(img_copy, (1024, 1024), interpolation=cv2.INTER_CUBIC)
                cv2.imshow(' ', img_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            ############################################################################################
            # clustering
            print('clustering...')

            dist_ls = np.array(list(dist_dict.values())).reshape(-1, 1)

            SSE = []
            for k in range(1, 9):
                model = KMeans(k)
                try:
                    model.fit(dist_ls)
                except ValueError:
                    break
                SSE.append(model.inertia_)
            # plt.plot(list(range(1, 9)), SSE)
            # plt.show()
            for i in range(len(SSE) - 1):
                if SSE[i] - SSE[i + 1] < 10:
                    k = i + 1
                    break
            # print(line_length)
            model = KMeans(k)
            try:
                model.fit(dist_ls)
            except ValueError:
                return (default_polygon, 0, line_length) if side else (
                    default_polygon, default_polygon, 0, 0, line_length)

            # print(model.cluster_centers_)
            # print('distance: ', max(model.cluster_centers_))
            dist_label = list(model.cluster_centers_).index(max(model.cluster_centers_))
            dist_labels = [i for i, _ in enumerate(list(model.labels_)) if _ == dist_label]

            ##################################################################################
            # show pair with road width
            img_copy = img.copy()
            for i in range(len(center_line[:-1])):
                cv2.line(
                    img_copy,
                    tuple(map(int, center_line[i])),
                    tuple(map(int, center_line[i + 1])),
                    (0, 0, 127),
                    thickness=5
                )
            for i in dist_labels:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pt1 = int(list(dist_dict.keys())[i][0])
                pt2 = int(list(dist_dict.keys())[i][1])
                for pt in [pt1, pt2]:
                    cv2.line(img_copy,
                             tuple(map(int, (df_segment_filter.loc[pt, 'x1'], df_segment_filter.loc[pt, 'y1']))),
                             tuple(map(int, (df_segment_filter.loc[pt, 'x2'], df_segment_filter.loc[pt, 'y2']))),
                             color,
                             thickness=3
                             )
            img_copy = cv2.resize(img_copy, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            if show_paired_segment:
                cv2.imshow(' ', img_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            ##########################################################################################
            # determine a mask
            # one side of center line

            line1_dist = []
            line2_dist = []
            for i in dist_labels:
                lines = list(dist_dict.keys())[i]
                line1 = df_segment.loc[lines[0]]
                line2 = df_segment.loc[int(lines[1])]
                linestring1 = LineString([(line1.x1, line1.y1), (line1.x2, line1.y2)])
                linestring2 = LineString([(line2.x1, line2.y1), (line2.x2, line2.y2)])
                pt1 = Point((line1.x1 + line1.x2) / 2, (line1.y1 + line1.y2) / 2)
                pt2 = Point((line2.x1 + line2.x2) / 2, (line2.y1 + line2.y2) / 2)
                if selected_area1.intersects(linestring1) and not selected_area1.intersects(linestring2):
                    line1_dist.append(LineString(center_line).distance(pt1))
                    line2_dist.append(LineString(center_line).distance(pt2))
                elif selected_area1.intersects(linestring2) and not selected_area1.intersects(linestring1):
                    line1_dist.append(LineString(center_line).distance(pt2))
                    line2_dist.append(LineString(center_line).distance(pt1))
                elif selected_area1.intersects(linestring1):
                    line1_dist.append(max(LineString(center_line).distance(pt1), LineString(center_line).distance(pt2)))
                    line2_dist.append(min(LineString(center_line).distance(pt1), LineString(center_line).distance(pt2)))
                else:
                    line2_dist.append(max(LineString(center_line).distance(pt1), LineString(center_line).distance(pt2)))
                    line1_dist.append(min(LineString(center_line).distance(pt1), LineString(center_line).distance(pt2)))
            line1_dist = sum(line1_dist) / len(line1_dist)
            line2_dist = sum(line2_dist) / len(line2_dist)

            mask1 = LineString(center_line).buffer(signal * line1_dist, cap_style=2, join_style=2, single_sided=True)
            mask2 = LineString(center_line).buffer(-signal * line2_dist, cap_style=2, join_style=2, single_sided=True)

            img_copy = img.copy()
            mask1_buffer = list(
                zip(map(int, mask1.exterior.coords.xy[0]), map(int, mask1.exterior.coords.xy[1])))
            mask2_buffer = list(
                zip(map(int, mask2.exterior.coords.xy[0]), map(int, mask2.exterior.coords.xy[1])))

            # draw selected area
            cv2.drawContours(
                img_copy,
                [np.array(mask1_buffer)],
                -1,
                (68, 140, 204),
                thickness=cv2.FILLED
            )

            if not side:
                cv2.drawContours(
                    img_copy,
                    [np.array(mask2_buffer)],
                    -1,
                    (68, 140, 204),
                    thickness=cv2.FILLED
                )
            for i in range(len(center_line[:-1])):
                cv2.line(
                    img_copy,
                    tuple(map(int, center_line[i])),
                    tuple(map(int, center_line[i + 1])),
                    (0, 0, 127),
                    thickness=5
                )
            cv2.rectangle(
                img_copy,
                (20, 20),
                (600, 100),
                (255, 255, 255),
                cv2.FILLED
            )
            cv2.putText(
                img_copy,
                'road length: {} m'.format(int(line_length)),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 0),
                thickness=2
            )
            cv2.putText(
                img_copy,
                'road width:  {} m'.format(round(max(model.cluster_centers_)[0], 1)),
                (50, 90),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 0),
                thickness=2
            )
            for i in dist_labels:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pt1 = int(list(dist_dict.keys())[i][0])
                pt2 = int(list(dist_dict.keys())[i][1])
                for pt in [pt1, pt2]:
                    cv2.line(img_copy,
                             tuple(map(int, (df_segment_filter.loc[pt, 'x1'], df_segment_filter.loc[pt, 'y1']))),
                             tuple(map(int, (df_segment_filter.loc[pt, 'x2'], df_segment_filter.loc[pt, 'y2']))),
                             color,
                             thickness=3
                             )

            cv2.addWeighted(
                img_copy, 0.5, img, 0.5, 0, img_copy
            )
            img_copy = cv2.resize(img_copy, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(mask_name, img_copy)
            if show_mask:
                cv2.imshow(' ', img_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # draw mask
            cv2.drawContours(
                img_mask,
                [np.array(mask1_buffer)],
                -1,
                color,
                thickness=cv2.FILLED
            )
            if not side:
                cv2.drawContours(
                    img_mask,
                    [np.array(mask2_buffer)],
                    -1,
                    color,
                    thickness=cv2.FILLED
                )
            for i in range(len(center_line[:-1])):
                cv2.line(
                    img,
                    tuple(map(int, center_line[i])),
                    tuple(map(int, center_line[i + 1])),
                    (0, 0, 127),
                    thickness=5
                )
            return (mask1, round(line1_dist * resolution, 2), round(line_length, 2)) if side else \
                (mask1, mask2, round(line1_dist * resolution, 2), round(line2_dist * resolution, 2),
                 round(line_length, 2))

        def get_selected_area(signal):
            selected_area = LineString(center_line).buffer(
                signal * offset, cap_style=2, join_style=2, single_sided=True
            ).union(
                LineString(center_line).buffer(-signal * 3.5 / resolution, cap_style=2, join_style=2, single_sided=True)
            ).difference(
                boundary_area).intersection(Polygon([(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)]))
            selected_area1 = LineString(center_line).buffer(signal * offset, cap_style=2, join_style=2,
                                                            single_sided=True).difference(boundary_area)
            return selected_area, selected_area1

        if geo_info.CARTO == 4 and not geo_info.ONEWAY:
            selected_area, selected_area1 = get_selected_area(1)
            mask1, road_width1, road_length = get_mask(selected_area, selected_area1, '_1')
            selected_area, selected_area1 = get_selected_area(-1)
            mask2, road_width2, road_length = get_mask(selected_area, selected_area1, '_0', -1)
        else:
            selected_area = LineString(center_line).buffer(offset, cap_style=2, join_style=2).difference(
                boundary_area).intersection(Polygon([(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)]))
            selected_area1 = LineString(center_line).buffer(offset, cap_style=2, join_style=2,
                                                            single_sided=True).difference(boundary_area)
            mask1, mask2, road_width1, road_width2, road_length = get_mask(selected_area, selected_area1, '')
        # write to file
        if not geo_info.ONEWAY:
            gdf_line.loc[line_index, 'width_1'] = road_width1
            gdf_line.loc[line_index, 'width_0'] = road_width2
        else:
            gdf_line.loc[line_index, 'width_1'] = road_width1
            gdf_line.loc[line_index, 'width_0'] = road_width2
        gdf_line.loc[line_index, 'mask'] = mask1 if mask2 == default_polygon else \
            mask2 if mask1 == default_polygon else mask1.union(mask2)
        gdf_line.loc[line_index, 'width'] = road_width1 + road_width2
        gdf_line.loc[line_index, 'length'] = road_length

    cv2.addWeighted(
        img_mask, 0.5, img, 0.5, 0, img_mask
    )
    img_mask = cv2.resize(img_mask, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(file_path, 'mask.png'), img_mask)
    # cv2.imshow(' ', img_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # gdf_line.geometry = gdf_line.geometry.apply(lambda x: Polygon([(0, 0), (0, 0), (0, 0)]) if type(x) == LineString else x)
    # gdf_line_geo = gpd.GeoDataFrame(
    #     gdf_line, geometry='geometry'
    # )

    gdf_line.to_csv(geo_file)
    return gdf_line


if __name__ == '__main__':
    # parameters
    # latitude, longitude = 43.72101, -79.3296  # highway
    # latitude, longitude = 43.7415, -79.3304  # curve ramp
    # latitude, longitude = 43.63792, -79.4627  # highway
    # latitude, longitude = 43.63881, -79.4566  # highway 和上一个类似
    # latitude, longitude = 43.63111, -79.4289  # lake shore
    # latitude, longitude = 43.63164, -79.4317  # lake shore & gardiner
    # latitude, longitude = 43.63247, -79.43  # highway
    # latitude, longitude = 43.63319, -79.4097  # major road, 《curve》
    # latitude, longitude = 43.63545, -79.4416  # highway
    # latitude, longitude = 43.65912, -79.3544  # DVP, 《有带树小路》
    # latitude, longitude = 43.70924, -79.3343  # main road
    # latitude, longitude = 43.71699, -79.3267  # highway <树小路>
    # latitude, longitude = 43.72863, -79.3306  # highway
    # latitude, longitude = 43.74006, -79.334  # 环岛
    # latitude, longitude = 43.75289, -79.363  # 有树大路
    # latitude, longitude = 43.63606, -79.401  # bathurst-lakeshore <带树小路， doubleway>
    # latitude, longitude = 43.6485, -79.3588  # gardiner
    # latitude, longitude = 43.64095, -79.3822  # gardiner
    # latitude, longitude = 43.64265, -79.3788  # gardiner
    # latitude, longitude = 43.68652, -79.3347  # main road
    #################################################################
    # latitude, longitude = 43.6458, -79.3695  # gardiner
    # latitude, longitude = 43.63032, -79.4204  # 效果不好的特例 lakeshore
    latitude, longitude = 43.63736, -79.4096  # gardiner
    # latitude, longitude = 43.63903, -79.3898  # gardiner? 是不是桥阿
    # latitude, longitude = 43.63996, -79.3845  # gardiner
    # latitude, longitude = 43.64857, -79.3589  # gardiner
    # latitude, longitude = 43.77303, -79.443  # main road, 还行
    # latitude, longitude = 43.78293, -79.2664  # main road
    # latitude, longitude = 43.79427, -79.2393  # main road
    main(latitude, longitude,
         img_scale='2048*2048*2*19',
         show_filtered_segment=False,
         show_paired_segment=False,
         show_selected_area=False,
         show_mask=True
         )

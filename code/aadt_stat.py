# Python vision: Python3.5
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:


import pandas as pd
import os
from datetime import datetime
import numpy as np
import re

# path definition
path_station = '/home/ming/Desktop/Satellite/data/AADT_per_station.csv'
path_PRTCS = '/media/ming/data/PRTCS'
path_output = '/home/ming/Desktop/Satellite/data/aadt_stat_hourly.csv'

# read the file containing permanent stations coordinate, google maps image captured date and time
df_station = pd.read_csv(path_station, usecols=list(range(10)))

# build a new table
year_ls = list(range(2006, 2017))
dir_ls = ['positive', 'negative']
col_year_dir = list(map(lambda x: '{}_{}'.format(x[0], x[1][0]), zip(sorted(year_ls * 2), dir_ls * len(year_ls))))
cols = ['station_id', 'latitude', 'longitude', 'datetime_maps', 'ave', 'ave_p', 'ave_n'] \
       + list(map(str, year_ls)) + col_year_dir
df_target = pd.DataFrame(columns=cols)


def calib_time(time_maps):
    '''
    convert the google map image time to quarter times, like 00, 15, 30, 45
    :param time_maps:
    :return: str, HHMM
    '''
    time_maps = str(int(time_maps)).zfill(4)
    hour = int(time_maps[:2])
    minute = int(time_maps[2:])
    if minute < 15:
        minute = 0
    elif minute < 30:
        minute = 15
    elif minute < 45:
        minute = 30
    else:
        minute = 45
    return str(hour).zfill(2) + str(minute).zfill(2)


def fill_count(df_target, row, hourly=True):
    '''
    fill the new table

    :param df_target: new table
    :param row: the row in dataframe stands for info of a permanent station
    :param hourly: if True, sum the hourly count
    :return: None
    '''

    # the following params will be used in the search and calculation
    test = df_station.iloc[row]
    station_id = str(int(test['lroad id']))
    date_maps = str(int(test['date of maps']))
    time_maps = test['time']
    latitude = test['y.1']
    longitude = test['x.1']

    # convert the google map image time to quarter times
    time_maps = calib_time(time_maps)
    # convert google map image time to format datetime
    datetime_maps = datetime.strptime('{} {}'.format(date_maps, time_maps), '%Y%m%d %H%M')
    # assign part of values to the new table
    df_target.loc[row, ['station_id', 'latitude', 'longitude', 'datetime_maps']] = [station_id, latitude, longitude,
                                                                                    datetime_maps]

    # look through files for all years and both direction
    for year in year_ls:
        for direction in dir_ls:
            print(year, direction, station_id)

            path_dir_year = os.path.join(path_PRTCS, direction, '15min_counts_{}'.format(year))
            file_ls = os.listdir(path_dir_year)
            try:
                # find out the txt file for a station according to the station id
                file_name = [_ for _ in file_ls if station_id in _.split('_')[0]][0]
            except IndexError:
                # if there is no file for this year and this direction, skip to other year/direction
                continue

            file_name = os.path.join(path_dir_year, file_name)
            try:
                # read the txt file as a table
                df_txt = pd.read_csv(file_name, sep='\t', header=None)
            except pd.errors.ParserError as e:
                # a few of txt files contains lines which is unable to read, skip the lines
                print(e)
                msg = str(e)
                line = re.findall(r'line \d+', msg)[0].split(' ')[1]
                df_txt = pd.read_csv(file_name, sep='\t', header=None, nrows=int(line) - 1)
            # convert the time column to format datatime
            df_txt[3] = pd.to_datetime(df_txt[3])
            # the hourly count
            if hourly:
                # sum up the count of quarters
                datetime_filtered = list(
                    map(lambda x: datetime.strptime('{} {} {}{}'.format(
                        year, date_maps[4:], time_maps[:2], x), '%Y %m%d %H%M'),
                        ['00', '15', '30', '45']))
                veh_count = df_txt[df_txt[3].isin(datetime_filtered)][4].values
                veh_count = np.nan if len(veh_count) == 0 else sum(veh_count)
                # write to the new table
                df_target.loc[row, '{}_{}'.format(year, direction[0])] = veh_count
            # the 15min count
            else:
                datetime_filtered = datetime.strptime('{} {} {}'.format(year, date_maps[4:], time_maps), '%Y %m%d %H%M')
                try:
                    veh_count = df_txt[df_txt[3] == datetime_filtered][4].values[0]
                except IndexError:
                    continue
                else:
                    df_target.loc[row, '{}_{}'.format(year, direction[0])] = veh_count


# fill the new table for each permanent station
for row in df_station.index:
    fill_count(df_target, row)
    # df_target.to_csv(path_output)

# sum up counts of both directions for each year
for year in year_ls:
    df_target[str(year)] = df_target['{}_p'.format(year)] + df_target['{}_n'.format(year)]

# average yearly count
df_target['ave'] = df_target[list(map(str, year_ls))].mean(axis=1)
# average yearly count for single direction
df_target['ave_p'] = df_target[[_ for _ in col_year_dir if 'p' in _]].mean(axis=1)
df_target['ave_n'] = df_target[[_ for _ in col_year_dir if 'n' in _]].mean(axis=1)
# write
df_target.to_csv(path_output)

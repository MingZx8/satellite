# Python version: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:

from PIL import Image
import os


# png to tiff
def png2tiff(path, path_dest):
    for file in os.listdir(path):
        print(file)
        img = Image.open(os.path.join(path, file))
        img.save(
            os.path.join(
                path_dest, file[:-4] + '.tiff'
            )
        )


def png2pgm(input, output):
    img = Image.open(input)
    img = img.convert('L')
    print(img.size)
    # img = img.resize(img.size, Image.BILINEAR)
    img.save(
        output
    )


def generate(
        img_path='/media/ming/data/google_map_imgs/image/image.png',
        output_path='/media/ming/data/google_map_imgs/',
        exe='/home/ming/Desktop/Satellite/code/lsd_1.6/lsd',
):
    output = '{}/lsd.pgm'.format(output_path)
    png2pgm(img_path, output)

    result = '{}/lsd.txt'.format(output_path)
    os.system(
        '{} {} {}'.format(exe, output, result)
    )


# def generate(lat, lon, img_scale,
#              path='/media/ming/data/google_map_imgs',
#              exe='/home/ming/Desktop/Satellite/code/lsd_1.6/lsd',
#              centreline_label=None):
#     file_name = '{}'.format(centreline_label) if centreline_label else '{},{}'.format(lat, lon)
#     input = '{}/{}/{}/image/image.png'.format(path, img_scale, file_name)
#     output = '{}/{}/{}/lsd.pgm'.format(path, img_scale, file_name)
#     png2pgm(input, output)
#
#     result = '{}/{}/{}/lsd.txt'.format(path, img_scale, file_name)
#     os.system(
#         '{} {} {}'.format(exe, output, result)
#     )


if __name__ == '__main__':
    latitude, longitude = 43.79427, -79.2393
    generate()

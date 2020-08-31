# Python version: Python3.6
# @Author: MingZZZZZZZZ
# @Date created: 2020
# @Date modified: 2020
# Description:

from PIL import Image
import os

exe = '/home/ming/Desktop/Satellite/code/lsd_1.6/lsd'


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
    img.save(
        output
    )


def generate(file_path):
    print('detecting line segments...')
    img_path = os.path.join(file_path, 'image', 'image.png')
    output = '{}/lsd.pgm'.format(file_path)
    png2pgm(img_path, output)

    result = '{}/lsd.txt'.format(file_path)
    os.system(
        '{} {} {}'.format(exe, output, result)
    )
    os.remove(output)


if __name__ == '__main__':
    file_path = '../output/eg'
    generate(file_path)

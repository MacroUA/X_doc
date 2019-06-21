from os import listdir
from PIL import Image


files_dir = './input/lis_raw/'
output_dir = './input/lis/'

files = listdir(files_dir)

for fi in files:
    im = Image.open(files_dir+fi)
    width, height = im.size
    print(fi, width, height)
    #im.show()

    if width != height:
        if width > height:
            dy = 0
            dx = (width - height) // 2
        else:
            dx = 0
            dy = (height - width) // 2

        x1 = 0 + dx
        y1 = 0 + dy
        x2 = width - dx
        y2 = height - dy
        im = im.crop((x1, y1, x2, y2))

    print(fi, im.size)

    #im.show()
    #exit()

    im.save(output_dir + fi)
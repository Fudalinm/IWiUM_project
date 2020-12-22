import re
import glob
from PIL import Image
import itertools
import imageio
from imgaug import augmenters as iaa
from pathlib import Path

NEW_DIM = (130, 100)
NEW_ZOOM = 500
DATA_PATH = './data/'
LOW_RM = DATA_PATH + 'LowRm/'
HIGH_RM = DATA_PATH + 'HighRm/'

TRANSFORMED = './transformed_zoom{}_dim{}x{}/'.format(NEW_ZOOM, NEW_DIM[0], NEW_DIM[1])
TRANSFORMED_NORMALIZED = TRANSFORMED + 'normalized/'
TRANSFORMED_AUGMENTED = './transformed_augmented_zoom{}_dim{}x{}/'.format(NEW_ZOOM, NEW_DIM[0], NEW_DIM[1])


# TODO: It should clear data which does not have specified zoom in its name
def clear_data():
    pass


def normalize_data(data):
    out = []
    for d in data:
        p, im, res, zoom = d
        add_to_out = []
        if zoom < NEW_ZOOM:
            # we can sample more images
            range_res = int(res[0] / (NEW_ZOOM / zoom)), int(res[1] / (NEW_ZOOM / zoom))
            centre_x = [x for x in range(range_res[0], res[0] - range_res[0], range_res[0])]
            centre_y = [x for x in range(range_res[1], res[1] - range_res[1], range_res[1])]

            centre = list(itertools.product(centre_x, centre_y))

            i = 0
            for c in centre:
                start_res = c[0] - range_res[0], c[1] - range_res[1]
                end_res = c[0] + range_res[0], c[1] + range_res[1]
                area = (start_res[0], start_res[1], end_res[0], end_res[1])
                o_im = im.crop(area)
                o_zoom = NEW_ZOOM
                o_p = TRANSFORMED + p[:-4] + "_" + str(i) + '.png'
                i += 1
                o_res = NEW_DIM
                o_im = o_im.resize(NEW_DIM)
                add_to_out.append([o_p, o_im, o_res, o_zoom])

        else:
            o_p = TRANSFORMED + p[:-4] + '.png'
            o_im = im  # .convert('LA')
            o_zoom = zoom
            o_res = NEW_DIM
            o_im = o_im.resize(NEW_DIM)
            add_to_out.append([o_p, o_im, o_res, o_zoom])

        out += add_to_out

    return out


def save_data(data):
    Path(TRANSFORMED).mkdir(parents=True, exist_ok=True)
    Path(TRANSFORMED + LOW_RM).mkdir(parents=True, exist_ok=True)
    Path(TRANSFORMED + HIGH_RM).mkdir(parents=True, exist_ok=True)

    Path(TRANSFORMED_AUGMENTED).mkdir(parents=True, exist_ok=True)
    Path(TRANSFORMED_AUGMENTED + LOW_RM).mkdir(parents=True, exist_ok=True)
    Path(TRANSFORMED_AUGMENTED + HIGH_RM).mkdir(parents=True, exist_ok=True)

    for o_p, o_im, res, o_zoom in data:
        o_im.save(o_p, 'PNG')


# TODO: add data augmentation
# This function returns only augmented data
def augment_data(data):
    rotate = iaa.Affine(rotate=(-25, 25))
    gaussian_noise = iaa.AdditiveGaussianNoise(scale=(10, 60))
    contrast = iaa.GammaContrast(gamma=2.0)

    # https://towardsdatascience.com/data-augmentation-techniques-in-python-f216ef5eed69
    out = []
    for p, _, res, zoom in data:
        im = imageio.imread(p)

        # rotate
        x = p.split("/")
        x[1] = TRANSFORMED_AUGMENTED
        x[-1] = x[-1][:-4] + "_rotation" + x[-1][-4:]
        r_p = "/".join(x)
        r_im = Image.fromarray(rotate.augment_image(im))

        out.append([r_p, r_im, res, zoom])

        # gamma/brigtnes
        x = p.split("/")
        x[1] = TRANSFORMED_AUGMENTED
        x[-1] = x[-1][:-4] + "_gamma" + x[-1][-4:]
        g_p = "/".join(x)
        g_im = Image.fromarray((contrast.augment_image(im)))

        out.append([g_p, g_im, res, zoom])

        # gausian
        x = p.split("/")
        x[1] = TRANSFORMED_AUGMENTED
        x[-1] = x[-1][:-4] + "_gaussian" + x[-1][-4:]
        gau_p = "/".join(x)
        gau_im = Image.fromarray(gaussian_noise.augment_image(im))

        out.append([gau_p, gau_im, res, zoom])

    return out


def find_zoom(s):
    patter = '\d+x'
    r = re.findall(patter, s)
    return int(r[-1][:-1])


def file_with_resolution(s):
    im = Image.open(s)
    return im, im.size


def load_data():
    low = glob.glob(LOW_RM + '*.jpg')
    high = glob.glob(HIGH_RM + '*.jpg')

    low_with_zoom = [[x, file_with_resolution(x)[0], file_with_resolution(x)[1], find_zoom(x)] for x in low]
    high_with_zoom = [[x, file_with_resolution(x)[0], file_with_resolution(x)[1], find_zoom(x)] for x in high]

    return low_with_zoom, high_with_zoom


def find_bounds(l):
    zoom = (min(l, key=lambda x: x[3])[3], max(l, key=lambda x: x[3])[3])
    width = (min(l, key=lambda x: x[2][0])[2][0], max(l, key=lambda x: x[2][0])[2][0])
    height = (min(l, key=lambda x: x[2][1])[2][1], max(l, key=lambda x: x[2][1])[2][1])

    return {'category': ('min', 'max'), 'zoom': zoom, 'width': width, 'height': height}


if __name__ == '__main__':
    clear_data()
    low, high = load_data()
    print(low[0])
    print(find_bounds(low + high))

    normalized_low = normalize_data(low)
    normalized_high = normalize_data(high)

    save_data(normalized_low + normalized_high)

    augmented_low = augment_data(normalized_low)
    augmented_high = augment_data(normalized_high)

    save_data(augmented_low + augmented_high)

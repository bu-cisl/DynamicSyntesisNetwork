import time
from typing import List, Iterable, Dict
import numpy as np
import scipy.io as sio
import cv2

# consts
AUGMENTATION_N = 1
PATCH_STRIDE_TEST = 128
PATCH_STRIDE_TRAIN = 64
PATCH_SIZE = (128,128,100)

class Data:
    """
    Class to assist in handling 3D data
    """
    # consts
    PREPROCESS_THRESHOLD = 184
    # preprocess change rule table
    _CHANGE_TABLE = np.arange(256, dtype=float)
    _CHANGE_TABLE[PREPROCESS_THRESHOLD:256] = PREPROCESS_THRESHOLD
    _CHANGE_TABLE = ((_CHANGE_TABLE / PREPROCESS_THRESHOLD) ** 2 * 255).astype('uint8')

    # type hint
    _is_test = ...  # type: bool
    index_list = ...  # type: List[List[slice]]
    shape = ...  # type: tuple

    def __init__(self, fn: str, *, is_test: bool):
        """
        Read the 3D image data (.tiff format) for test and train. One object
        may include several 3D images according to img_num.

        :param data_i: Index of data (used to locate directory)
        :param is_test: Whether it is for test (or not = for training)
        :param data_dir: Actual directory for data
        """
        self._is_test = is_test
        self._img_num = 1

        # Read data

        self._o_images = []
        self._g_images = []
        self._h_images = []

        ###################################################
        # read backprop
        filename = '%s/f_bprop_abs_01.mat'%(fn)
        mat_contents = sio.loadmat(filename)
        temp_o = mat_contents['f_bprop_abs_01']
        self._set_or_check_shape(temp_o)
        self._o_images.append(temp_o)

        ###################################################
        # read 3D gt
        filename = '%s/obj_maxproj.tif' % (fn)
        #print('  ' + filename)
        temp_g = np.stack(cv2.imreadmulti(filename, flags=0)[1],axis=2)
        temp_g = temp_g.astype(np.float32)
        temp_g[temp_g > 0] = 1
        self._set_or_check_shape(temp_g)
        self._g_images.append(temp_g)

        ###################################################
        # read hologram
        filename = '%s/holo.mat' % (fn)
        mat_contents = sio.loadmat(filename)
        temp_h = mat_contents['meas']
        self._h_images.append(temp_h)

        ###################################################
        # init index list
        self.index_list = []
        nx, ny, nz = self.shape
        pnx, pny, pnz = PATCH_SIZE

        if self._is_test:
            ps = PATCH_STRIDE_TEST
            offset = 128
        else:
            ps = PATCH_STRIDE_TRAIN
            offset = 0

        psz = 64
        for i in range(0, nx - ps + offset, ps):
            for j in range(0, ny - ps + offset, ps):
                for k in range(0, nz - psz + 50, psz):
                    idx = []
                    # nx
                    if i < nx - pnx:
                        idx.append(slice(i, i + 128))
                    else:
                        idx.append(slice(nx - 128, nx))
                    # ny
                    if j < ny - pny:
                        idx.append(slice(j, j + 128))
                    else:
                        idx.append(slice(ny - 128, ny))
                    # nz
                    if k < nz - pnz:
                        idx.append(slice(k, k + 100))
                    else:
                        idx.append(slice(nz - 100, nz))

                    self.index_list.append(idx)

    def _set_or_check_shape(self, img: np.ndarray):
        if not isinstance(self.shape, tuple):
            self.shape = img.shape
        elif self.shape != img.shape:
            raise ValueError("An image have different shape " + str(img.shape) +
                             " with shape of other images " + str(self.shape) + " !")

    def get_original_image(self, img_n: int) -> np.ndarray:
        return self._o_images[img_n]

    def get_ground_truth(self, img_n: int) -> np.ndarray:
        return self._g_images[img_n]

    def get_holo(self, img_n: int) -> np.ndarray:
        return self._h_images[img_n]

    def __len__(self):
        return self._img_num

    def __getitem__(self, item):
        return {'original_image': self.get_original_image(item),
                'ground_truth': self.get_ground_truth(item)}


class Patch:
    """
    An image patch. Including a raw image and a label.

    Default shape: 128x128x128
    """
    def __init__(self, data: Data, img_n: int, index: List[slice]):
        self._data = data
        self._index = index
        self._img_n = img_n

    def get_index(self):
        return self._index

    def get_data(self):
        return self._data

    def get_original_image(self):
        return self._data.get_original_image(self._img_n)[self._index]

    def get_ground_truth(self):
        return self._data.get_ground_truth(self._img_n)[self._index]

    def get_holo(self):
        return self._data.get_holo(self._img_n)[self._index[0:2]]


class Batch:
    """
    A batch of data in numpy.array. Axis order is 'NDHWC'
    """
    def __init__(self, patch_list: List[Patch]):
        """
        Stack at batch axis and then, add "channels" axis for original image and
        (Changed:stack 2 channels) for ground truth.
        Axis order "NDHWC": [batch, depth, height, width, channels]

        :param patch_list: list of patches in this batch
        """
        self._o = np.stack([patch.get_original_image() for patch in patch_list])[..., np.newaxis]
        g_temp = np.stack([patch.get_ground_truth() for patch in patch_list])
        # self._g = np.stack((g_temp, 1 - g_temp), axis=-1)
        self._g = g_temp[..., np.newaxis]
        assert self._o.shape[:-1] == self._g.shape[:-1]
        self._p = patch_list

        self._h = np.stack([patch.get_holo() for patch in patch_list])[..., np.newaxis]

    def get_original_images_in_batch(self):
        return self._o

    def get_ground_truths_in_batch(self):
        return self._g

    def get_holos_in_batch(self):
        return self._h

    def get_data_list(self):
        return [patch.get_data() for patch in self._p]

    def get_index_list(self):
        return [patch.get_index() for patch in self._p]


def batch_generator(data_list: List[Data], batch_size: int) -> Iterable[Batch]:
    """
    A generator of image patches. Get image from disk (.tiff files) and
    return small patches to input into TensorFlow
    (Written by Jiabei Zhu on Jul. 30 2018)

    :param data_list: data source of data
    :param batch_size: the number of patches in a batch
    :return: dictionary of {"originalImage":'NDHWC' data in numpy array,
        "groundTruth":(same format)}
    """

    # get random list
    yield_sequence = [
        Patch(i, j, k)
        for i in data_list for k in i.index_list
        for j in range(len(i))  # test only original now
    ]

    np.random.shuffle(yield_sequence)

    # do yield
    patch_list = []
    for patch in yield_sequence:  # type: Patch
        patch_list.append(patch)
        if len(patch_list) == batch_size:
            yield Batch(patch_list)
            patch_list = []

def save_3d_tiff(path_prefix: str, img_dict: Dict[str, np.ndarray]):
    if path_prefix[-1] != '/':
        path_prefix = path_prefix + '/'
    for name in img_dict:
        img = img_dict[name]
        if img.min() >= 0 and img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            raise ValueError('The img data is not in 0~1!')
        if len(img.shape) != 3:
            raise ValueError('Images should be 3D numpy.ndarray')
        img = img.transpose((2, 0, 1))
        if name[-5:] != '.tiff' or name[-4:] != '.tif':
            name = name + '.tiff'
        
        import imageio
        imageio.mimwrite(path_prefix + name, img)
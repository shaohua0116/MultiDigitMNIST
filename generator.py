import subprocess
import os
import os.path as osp
import numpy as np
from imageio import imwrite
import argparse

mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']


def check_mnist_dir(data_dir):

    downloaded = np.all([osp.isfile(osp.join(data_dir, key)) for key in mnist_keys])
    if not downloaded:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_mnist(data_dir)
    else:
        print('MNIST was found')


def download_mnist(data_dir):

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    for k in mnist_keys:
        k += '.gz'
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gunzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)


def extract_mnist(data_dir):

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1))

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((num_mnist_train)))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1))

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((num_mnist_test)))

    return np.concatenate((train_image, test_image)), \
        np.concatenate((train_label, test_label))


def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int)


def generator(config):
    # check if mnist is downloaded. if not, download it
    check_mnist_dir(config.mnist_path)

    # extract mnist images and labels
    image, label = extract_mnist(config.mnist_path)
    h, w = image.shape[1:3]

    # split: train, val, test
    rs = np.random.RandomState(config.random_seed)
    num_original_class = len(np.unique(label))
    num_class = len(np.unique(label))**config.num_digit
    classes = list(np.array(range(num_class)))
    rs.shuffle(classes)
    num_train, num_val, num_test = [
            int(float(ratio)/np.sum(config.train_val_test_ratio)*num_class)
            for ratio in config.train_val_test_ratio]
    train_classes = classes[:num_train]
    val_classes = classes[num_train:num_train+num_val]
    test_classes = classes[num_train+num_val:]

    # label index
    indexes = []
    for c in range(num_original_class):
        indexes.append(list(np.where(label == c)[0]))

    # generate images for every class
    assert config.image_size[1]//config.num_digit >= w
    np.random.seed(config.random_seed)

    if not os.path.exists(config.multimnist_path):
        os.makedirs(config.multimnist_path)

    split_classes = [train_classes, val_classes, test_classes]
    count = 1
    for i, split_name in enumerate(['train', 'val', 'test']):
        path = osp.join(config.multimnist_path, split_name)
        print('Generat images for {} at {}'.format(split_name, path))
        if not os.path.exists(path):
            os.makedirs(path)
        for j, current_class in enumerate(split_classes[i]):
            class_str = str(current_class)
            class_str = '0'*(config.num_digit-len(class_str))+class_str
            class_path = osp.join(path, class_str)
            print('{} (progress: {}/{})'.format(class_path, count, len(classes)))
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            for k in range(config.num_image_per_class):
                # sample images
                digits = [int(class_str[l]) for l in range(config.num_digit)]
                imgs = [np.squeeze(image[np.random.choice(indexes[d])]) for d in digits]
                background = np.zeros((config.image_size)).astype(np.uint8)
                # sample coordinates
                ys = sample_coordinate(config.image_size[0]-h, config.num_digit)
                xs = sample_coordinate(config.image_size[1]//config.num_digit-w,
                                       size=config.num_digit)
                xs = [l*config.image_size[1]//config.num_digit+xs[l]
                      for l in range(config.num_digit)]
                # combine images
                for i in range(config.num_digit):
                    background[ys[i]:ys[i]+h, xs[i]:xs[i]+w] = imgs[i]
                # write the image
                image_path = osp.join(class_path, '{}_{}.png'.format(k, class_str))
                # image_path = osp.join(config.multimnist_path, '{}_{}_{}.png'.format(split_name, k, class_str))
                imwrite(image_path, background)
            count += 1

    return image, label, indexes


def argparser():

    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mnist_path', type=str, default='./datasets/mnist/',
                        help='path to *.gz files')
    parser.add_argument('--multimnist_path', type=str, default='./datasets/multimnist')
    parser.add_argument('--num_digit', type=int, default=2)
    parser.add_argument('--train_val_test_ratio', type=int, nargs='+',
                        default=[64, 16, 20], help='percentage')
    parser.add_argument('--image_size', type=int, nargs='+',
                        default=[64, 64])
    parser.add_argument('--num_image_per_class', type=int, default=10000)
    parser.add_argument('--random_seed', type=int, default=123)
    config = parser.parse_args()
    return config


def main():

    config = argparser()
    assert len(config.train_val_test_ratio) == 3
    assert sum(config.train_val_test_ratio) == 100
    assert len(config.image_size) == 2
    generator(config)


if __name__ == '__main__':
    main()

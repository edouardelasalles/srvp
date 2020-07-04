# Adapted from https://github.com/edenton/svg/blob/master/data/convert_bair.py (cleaned, up-to-date).


import argparse
import os

import tensorflow as tf

from PIL import Image
from tensorflow.python.platform import gfile


def get_seq(data_dir, dname):
    """
    Enumerates videos from the original dataset.

    Parameters
    ----------
    data_dir : str
        Directory where original videos are saved.
    dname : stre
        'train' or 'test'. Determines whether training or testing videos should be processed.

    Yields
    -------
    str
        File name from which the video was extracted.
    int
        Video index in the file from which it was extracted.
    list
        List of PIL.Image.Image objects corresponding to frames of the video.
    """
    # Get list of video files
    data_dir = os.path.join(data_dir, 'softmotion30_44k', dname)
    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')
    # Enumerates videos (filename, index of file, list of images)
    for f in filenames:
        k = 0
        for serialized_example in tf.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_seq = []
            # Get all frames of the video
            for i in range(30):
                image_name = os.path.join(str(i), 'image_aux1', 'encoded')
                byte_str = example.features.feature[image_name].bytes_list.value[0]
                img = Image.frombytes('RGB', (64, 64), byte_str)
                image_seq.append(img)
            k = k + 1
            yield f, k, image_seq


def convert_data(data_dir, dname):
    """
    Preprocesses videos from the BAIR dataset in the input directory.

    Processed videos are saved in separate directories. They are of the form 'filename/index', where filename is the
    file from which the video was extracted, and index in the index of the video in that file. Each video frame is
    extracted in a .png file.

    Parameters
    ----------
    data_dir : str
        Directory where original videos are saved and processed videos will be saved.
    dname : str
        'train' or 'test'. Determines whether training or testing videos should be processed.
    """
    # Get videos from the original dataset
    seq_generator = get_seq(data_dir, dname)
    # Process videos
    for n, (f, k, seq) in enumerate(seq_generator):
        # Create a directory for the video
        f = os.path.splitext(os.path.basename(f))[0]
        dirname = os.path.join(data_dir, 'processed_data', dname, f, f'{k:03d}')
        os.makedirs(dirname)
        # Save all frames in .png files
        for i, img in enumerate(seq):
            img.save(os.path.join(dirname, f'{i:03d}.png'), 'PNG')
        print(f'{dirname} ({n + 1})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='BAIR preprocessing.',
        description='Generates training and testing videos for the BAIR dataset from the original videos, and stores \
                     them in folders `train` and `test` in the same directory. Each video frame is saved as a png \
                     file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where videos from the original dataset are stored.')
    args = parser.parse_args()

    print('Train sequences...')
    convert_data(args.data_dir, 'train')
    print(os.linesep)
    print('Test sequences...')
    convert_data(args.data_dir, 'test')

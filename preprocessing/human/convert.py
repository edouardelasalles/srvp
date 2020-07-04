# Copyright 2020 Mickael Chen, Edouard Delasalles, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import imageio
import os
import tqdm

import numpy as np

from PIL import Image


# Subjects used for training and testing videos
train_subjects = [1, 5, 6, 7, 8]
test_subjects = [9, 11]


def generate_from_mp4(data_dir, image_size, train=True):
    """
    Preprocesses videos from the Human3.6M dataset in the input directory.

    Processed videos are saved in the npz format and contain the following fields:
        - `image`: uint8 NumPy array of dimensions (length, width, height, channels);
        - `filename`: file name of the original video;
        - `subject`: subject identifier of the subject in the video.

    Parameters
    ----------
    data_dir : str
        Directory where original videos are saved and processed videos will be saved.
    image_size : int
        Width and height of the processed images.
    train : bool
        Determines whether training or testing videos should be processed. Subjects used for training and testing
        are determined by `train_subjects` and `test_subjects` defined in this file.
    """
    # Directory where the videos will be saved
    save_dir = os.path.join(data_dir, 'train' if train else 'test')
    # Create the directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Subjects to preprocess
    if train:
        subjects = train_subjects
    else:
        subjects = test_subjects

    progress_bar = tqdm.tqdm(total=120 * len(subjects), ncols=0)
    # Browse the list of subjects
    for subject_id in subjects:
        subject_dir = os.path.join(data_dir, f'S{subject_id}', 'Videos')
        # Browse the list of videos of the subject
        for video_file in os.listdir(subject_dir):
            if video_file[0] == '_':
                # If the video file name begins by `_ALL`, the video is ignored, as specified by authors of
                # "Unsupervised learning of object structure and dynamics from videos" (NeurIPS 2018)
                continue
                # Load video
            video_path = os.path.join(subject_dir, video_file)
            video = imageio.get_reader(video_path, 'ffmpeg')
            # Crop, then resize (sequentially, not simultaneously), as specified by authors of
            # "Unsupervised learning of object structure and dynamics from videos" (NeurIPS 2018)
            video_np = np.stack([
                np.array(Image.fromarray(frame).crop((100, 100, 900, 900)).resize((image_size, image_size),
                                                                                  resample=Image.LANCZOS))
                for frame in video.iter_data()
            ])

            # Save the processed video
            video_file_name = os.path.splitext(video_file)[0]  # Remove extension
            np.savez(
                os.path.join(save_dir, f'S{subject_id}-{video_file_name}'),
                image=video_np, filename=video_file_name, subject=subject_id
            )
            progress_bar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Human3.6M preprocessing.',
        description='Generates training and testing videos for the Human3.6M dataset from the original videos, and \
                     stores them in folders `train` and `test` in the same directory as npz files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where videos from the original dataset are stored.')
    parser.add_argument('--image_size', type=int, metavar='SIZE', default=64,
                        help='Width and height of resulting processed videos.')
    args = parser.parse_args()

    print('Train sequences...')
    generate_from_mp4(args.data_dir, args.image_size, train=True)
    print(os.linesep)
    print('Test sequences...')
    generate_from_mp4(args.data_dir, args.image_size, train=False)

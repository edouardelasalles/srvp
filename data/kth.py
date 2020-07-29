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


import os
from os.path import join

import numpy as np
from PIL import Image

from data.base import VideoDataset


class KTH(VideoDataset):
    """
    KTH dataset.

    Attributes
    ----------
    data : list
        List containing the dataset data. For KTH, it consists of a list of lists of image files, representing video
        frames.
    nx : int
        Width and height of the video frames.
    seq_len : int
        Number of frames to produce.
    train : bool
        Whether to use the training or testing dataset.
    """
    # List of actions in the dataset
    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

    def __init__(self, data, nx, seq_len, train):
        """
        Parameters
        ----------
        data : list
            List containing the dataset data. For KTH, it consists of a list of lists of image files, representing
            video frames.
        nx : int
            Width and height of the video frames.
        seq_len : int
            Number of frames to produce.
        train : bool
            Whether to use the training or testing dataset.
        """
        self.data = data
        self.nx = nx
        self.seq_len = seq_len
        self.train = train

    def change_seq_len(self, seq_len):
        """
        Changes the length of sequences in the dataset.

        Parameters
        ----------
        seq_len : int
            New sequence length.
        """
        self.seq_len = seq_len

    def _filter(self, data):
        return KTH(data, self.nx, self.seq_len, self.train)

    def __len__(self):
        if self.train:
            # Arbitrary number.
            # The number is a trade-off for max efficiency.
            # If too low, it is not good for batch size and multi-threaded dataloader.
            # If too high, it is not good for shuffling and sampling.
            return 500000
        return len(self.data)

    def __getitem__(self, index):
        if not self.train:
            # When testing, pick the selected video at its beginning (from the precomputed testing set)
            return self.data[index]
        # When training, pick a random video from the dataset, and extract a random sequence
        # Iterate until the selected video is long enough
        done = False
        while not done:
            vid_id = np.random.randint(len(self.data))
            vid = self.data[vid_id]
            vid_len = len(vid)
            if vid_len < self.seq_len:
                continue
            done = True
        # Random timestep for the beginning of the video
        t0 = np.random.randint(vid_len - self.seq_len + 1)
        # Extract the sequence from frame image files
        x = np.zeros((self.seq_len, self.nx, self.nx), dtype=np.uint8)
        for t in range(self.seq_len):
            img_path = vid[t0 + t]
            img = Image.open(img_path)
            # Only the first channel is used as this dataset is greyscale
            x[t] += np.array(img)[:, :, 0]
        # Returned video is an uint8 NumPy array of shape (length, width, height)
        return x

    @classmethod
    def make_dataset(cls, data_dir, nx, seq_len, train):
        """
        Creates a dataset from the directory where the dataset is saved.

        Parameters
        ----------
        data_dir : str
            Path to the dataset.
        nx : int
            Width and height of the video frames.
        seq_len : int
            Number of frames to produce.
        train : bool
            Whether to use the training or testing dataset.

        Returns
        -------
        data.kth.KTH
        """
        # Select the right fold (train / test)
        if train:
            # Loads all preprocessed videos
            data_dir = join(data_dir, f'processed_{nx}')
            data = []
            for c in cls.classes:
                for vid in os.listdir(join(data_dir, c)):
                    if not os.path.isdir(join(data_dir, c, vid)):
                        continue
                    # Removes the last five subjects for the testing set and keeps the first twenty subjects for the
                    # training set
                    person = int(vid.split('_')[0][-2:])
                    if person > 20:
                        continue
                    # Videos are lists of frame image files
                    images = sorted([
                        join(data_dir, c, vid, img)
                        for img in os.listdir(join(data_dir, c, vid)) if os.path.splitext(img)[1] == '.png'
                    ])
                    data.append(images)
        else:
            # If testing, load the precomputed dataset
            fname = f'svg_test_set_{seq_len}.npz'
            dataset = np.load(join(data_dir, fname), allow_pickle=True)
            sequences = dataset['sequences']
            data = [sequences[i] for i in range(len(sequences))]
        # Create and return the dataset
        return cls(data, nx, seq_len, train)

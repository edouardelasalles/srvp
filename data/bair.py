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

import numpy as np

from os.path import join
from PIL import Image

from data.base import VideoDataset


class Bair(VideoDataset):
    """
    BAIR dataset.

    Attributes
    ----------
    data : list
        List containing the dataset data. For BAIR, it consists of a list of lists of image files, representing video
        frames.
    nx : int
        Width and height of the video frames.
    nc : int
        Number of channels in the video data (3).
    seq_len : int
        Number of frames to produce.
    train : bool
        Whether to use the training or testing dataset.
    """
    def __init__(self, data, seq_len, train):
        """
        Parameters
        ----------
        data : list
            List containing the dataset data. For BAIR, it consists of a list of lists of image files, representing
            video frames.
        seq_len : int
            Number of frames to produce.
        train : bool
            Whether to use the training or testing dataset.
        """
        assert seq_len <= 30
        self.data = data
        self.nx = 64
        self.nc = 3
        self.seq_len = seq_len
        self.train = train

    def _filter(self, data):
        return Bair(data, self.seq_len, self.train)

    def __getitem__(self, index):
        vid = self.data[index]
        # Choose a random beginning for the video when training, otherwise select the beginning of the video
        t_0 = np.random.randint(30 - self.seq_len + 1) if self.train else 0
        # Load and group images
        x = np.zeros((self.seq_len, self.nx, self.nx, self.nc), dtype=np.uint8)
        for t in range(self.seq_len):
            x[t] += np.array(Image.open(vid[t_0 + t]))
        return x

    @classmethod
    def make_dataset(cls, data_dir, seq_len, train):
        """
        Creates a dataset from the directory where the dataset is saved.

        Parameters
        ----------
        data_dir : str
            Path to the dataset.
        seq_len : int
            Number of frames to produce.
        train : bool
            Whether to use the training or testing dataset.
        """
        # Select the right fold (train / test)
        if train:
            data_dir = join(data_dir, 'processed_data', 'train')
        else:
            data_dir = join(data_dir, 'processed_data', 'test')
        data = []
        i = 0
        # Store in data the videos
        for d1 in sorted(os.listdir(data_dir)):
            for d2 in sorted(os.listdir(join(data_dir, d1))):
                i += 1
                # Videos are lists of frame image files
                images = sorted([
                    join(data_dir, d1, d2, img)
                    for img in os.listdir(join(data_dir, d1, d2))
                    if os.path.splitext(img)[1] == '.png'
                ])
                data.append(images)
        # Create and return the dataset
        return cls(data, seq_len, train)

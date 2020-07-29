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

from data.base import VideoDataset


class Human(VideoDataset):
    """
    Human3.6M dataset.

    Loads the whole dataset into memory, thus leading to an important RAM usage.

    Attributes
    ----------
    data : list
        List of videos represented as uint8 NumPy arrays (length, width, height, channels).
    nx : int
        Width and height of the video frames.
    seq_len : int
        Number of frames to produce.
    train : bool
        Whether to use the training or testing dataset.
    subsampling : int
        Selects one in $subsampling frames from the original dataset.
    actual_seq_len : int
        Sequence length for the original dataset, taking into account intermediary frames that will be dropped in the
        output.
    """
    def __init__(self, data, nx, seq_len, subsampling, train):
        """
        Parameters
        ----------
        data : list
            List of videos represented as uint8 NumPy arrays (length, width, height, channels).
        nx : int
            Width and height of the video frames.
        seq_len : int
            Number of frames to produce.
        subsampling : int
            Selects one in $subsampling frames from the original dataset.
        train : bool
            Whether to use the training or testing dataset.
        """
        self.data = data
        self.nx = nx
        self.seq_len = seq_len
        self.train = train
        self.subsampling = subsampling
        self.actual_seq_len = (seq_len - 1) * subsampling + 1

    def change_seq_len(self, seq_len):
        """
        Changes the length of sequences in the dataset.

        Parameters
        ----------
        seq_len : int
            New sequence length.
        """
        self.seq_len = seq_len
        self.actual_seq_len = (seq_len - 1) * self.subsampling + 1

    def _filter(self, data):
        return Human(data, self.nx, self.seq_len, self.subsampling, self.train)

    def __len__(self):
        if self.train:
            # Arbitrary number.
            # The number is a trade-off for max efficiency
            # If too low, it is not good for batch size and multi-threaded dataloader
            # If too high, it is not good for shuffling and sampling
            return 500000
        return len(self.data)

    def __getitem__(self, index):
        if not self.train:
            # When testing, pick the selected video at its beginning (from the precomputed testing set)
            vid = self.data[index]
            t0 = 0
        else:
            # When training, pick a random video from the dataset, and extract a random sequence
            done = False
            # Iterate until the selected video is long enough
            while not done:
                vid_id = np.random.randint(len(self.data))
                vid = self.data[vid_id]
                vid_len = len(vid)
                if vid_len < self.actual_seq_len:
                    continue
                done = True
            # Random timestep for the beginning of the video
            t0 = np.random.randint(vid_len - self.actual_seq_len + 1)
        # Extract the sequence without the intermediary frames
        return vid[t0: t0 + self.actual_seq_len: self.subsampling]

    @classmethod
    def make_dataset(cls, data_dir, nx, seq_len, subsampling, train):
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
        subsampling : int
            Selects one in $subsampling frames from the original dataset.
        train : bool
            Whether to use the training or testing dataset.

        Returns
        -------
        data.human.Human
        """
        # Select the right fold (train / test)
        data_folder = os.path.join(data_dir, 'train' if train else f'test_set_{seq_len}_{subsampling}')
        data = []
        # Store in data the videos
        for video_file in sorted(os.listdir(data_folder)):
            video_path = os.path.join(data_folder, video_file)
            video_data = {k: v for k, v in np.load(video_path).items()}
            # Videos are uint8 NumPy arrays with shape (length, width, height, channels)
            data.append(video_data['image'])
        # Create and return the dataset
        return cls(data, nx, seq_len, subsampling, train)

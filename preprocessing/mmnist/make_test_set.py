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
import os

import numpy as np

from os.path import join
from tqdm import trange
from torchvision import datasets

from data.mmnist import MovingMNIST


if __name__ == "__main__":
    parser = argparse.ArgumentParser('''
        Generates the Moving MNIST testing set. Videos and latent space (position, speed) are saved in an npz file.
        ''')
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the testing set will be saved.')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=100,
                        help='Number of frames per testing sequences.')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    parser.add_argument('--deterministic', action='store_true',
                        help='If activated, generates a testing set for the deterministic version of the dataset.')
    parser.add_argument('--digits', type=int, metavar='NUM', default=2,
                        help='Number of digits to appear in each video.')
    parser.add_argument('--frame_size', type=int, metavar='SIZE', default=64,
                        help='Size of generated frames.')
    parser.add_argument('--max_speed', type=int, metavar='SPEED', default=4,
                        help='Maximum speed of generated trajectories.')
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(args.seed)

    # Load digits and shuffle them
    digits = datasets.MNIST(args.data_dir, train=False, download=True)
    digits_idx = np.random.permutation(len(digits))
    # Random trajectories are made using the dataset code
    trajectory_sampler = MovingMNIST([], args.frame_size, args.seq_len, args.max_speed, args.deterministic,
                                     args.digits, True)
    # Register videos, latent space (position, speed), labels of digits and digit images
    test_videos = []
    test_latents = []
    test_labels = []
    test_objects = []
    # The size of the testing set is the total number of testing digits in MNIST divided by the number of digits
    for i in trange(len(digits) // args.digits):
        x = np.zeros((args.seq_len, args.frame_size, args.frame_size), dtype=np.float32)
        latents = []
        labels = []
        objects = []
        # Pick the digits randomly chosen for sequence i and generate their trajectories
        for n in range(args.digits):
            img, label = digits[digits_idx[i * args.digits + n]]
            img = np.array(img, dtype=np.uint8)
            trajectory = trajectory_sampler._compute_trajectory(*img.shape)
            latents.append(np.array(trajectory))
            labels.append(label)
            objects.append(img)
            for t in range(args.seq_len):
                sx, sy, _, _ = trajectory[t]
                x[t, sx:sx + img.shape[0], sy:sy + img.shape[1]] += img
        x[x > 255] = 255
        test_videos.append(x.astype(np.uint8))
        test_latents.append(np.array(latents))
        test_labels.append(np.array(labels).astype(np.uint8))
        test_objects.append(np.array(objects))
    test_videos = np.array(test_videos, dtype=np.uint8).transpose(1, 0, 2, 3)
    test_latents = np.array(test_latents).transpose(2, 0, 1, 3)
    test_labels = np.array(test_labels, dtype=np.uint8)
    test_objects = np.array(test_objects)

    # Save results at the given path
    fname = f'mmnist_test_{args.digits}digits_{args.frame_size}.npz'
    if not args.deterministic:
        fname = f's{fname}'
    print(f'Saving testset at {join(args.data_dir, fname)}')
    # Create the directory if needed
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    np.savez_compressed(join(args.data_dir, fname),
                        sequences=test_videos, latents=test_latents, labels=test_labels, digits=test_objects)

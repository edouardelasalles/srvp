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

from tqdm import tqdm, trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Human3.6M testing set generation.',
        description='Generates the Human3.6M testing set from the testing videos by extracting fixed-length \
                     sequences. Videos are saved as npz files, like the training videos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the dataset is stored and the testing set will be saved.')
    parser.add_argument('--size', type=int, metavar='SIZE', default=1000,
                        help='Number of sequences to extract (size of the testing set).')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=53,
                        help='Number of frames per sequence to extract.')
    parser.add_argument('--subsampling', type=int, metavar='SUB', default=8,
                        help='Selects one in $SUB frames.')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(args.seed)

    # Directory where the videos will be saved
    save_dir = os.path.join(args.data_dir, f'test_set_{args.seq_len}_{args.subsampling}')
    # Create the directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f'Generating test set at {save_dir}...')

    video_files = sorted(os.listdir(os.path.join(args.data_dir, 'test')))
    nb_videos = len(video_files)
    actual_seq_len = (args.seq_len - 1) * args.subsampling + 1
    progress_bar = tqdm(total=args.size, ncols=0)
    # Randomly extract a determined number of videos
    for i in trange(args.size):
        # Randomly choose the video
        video_id = np.random.randint(nb_videos)
        video_path = os.path.join(args.data_dir, 'test', video_files[video_id])
        video_data = {k: v for k, v in np.load(video_path).items()}
        video_length = video_data['image'].shape[0]
        # Randomly choose the beginning of the video extract to be included in the testing set
        t_0 = np.random.randint(video_length - actual_seq_len + 1)
        # Intermediary frames are kept for compatibility purposes when the dataset in loaded in the code of the model
        video_data['image'] = video_data['image'][t_0: t_0 + actual_seq_len:]
        video_data['frame'] = t_0
        # Save the video
        np.savez(os.path.join(save_dir, f'test_{i}'), **video_data)

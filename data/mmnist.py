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

from torchvision import datasets

from data.base import VideoDataset


class MovingMNIST(VideoDataset):
    """
    Updated stochastic and deterministic MovingMNIST dataset, inspired by
    https://github.com/edenton/svg/blob/master/data/moving_mnist.py.

    See the paper for more information.

    Attributes
    ----------
    data : list
        When testing, list of testing videos represented as uint8 NumPy arrays (length, width, height).
        When training, list of digits shape to use when generating videos from the dataset.
    frame_size : int
        Width and height of the video frames.
    seq_len : int
        Number of frames to produce.
    max_speed : int
        Maximum speed of moving digits in the videos.
    deterministic : bool
        Whether to use the deterministic version of the dataset rather that the stochastic one.
    num_digits : int
        Number of digits in each video.
    train : bool
        Whether to use the training or testing dataset.
    eps : float
        Precision parameter to compute intersections between trajectories and frame borders.
    """
    eps = 1e-8

    def __init__(self, data, nx, seq_len, max_speed, deterministic, num_digits, train):
        """
        Parameters
        ----------
        data : list
            When testing, list of testing videos represented as uint8 NumPy arrays (length, width, height).
            When training, list of digits shape to use when generating videos from the dataset.
        nx : int
            Width and height of the video frames.
        seq_len : int
            Number of frames to produce.
        max_speed : int
            Maximum speed of moving digits in the videos.
        deterministic : bool
            Whether to use the deterministic version of the dataset rather that the stochastic one.
        num_digits : int
            Number of digits in each video.
        train : bool
            Whether to use the training or testing dataset.
        """
        self.data = data
        self.frame_size = nx
        self.seq_len = seq_len
        self.max_speed = max_speed
        self.deterministic = deterministic
        self.num_digits = num_digits
        self.train = train

    def _filter(self, data):
        return self.__class__(data, self.frame_size, self.seq_len, self.max_speed, self.deterministic,
                              self.num_digits, self.train)

    def __len__(self):
        if self.train:
            # Arbitrary number
            # The number is a trade-off for max efficiency
            # If too low, it is not good for batch size and multi threaded dataloader
            # If too high, it is not good for shuffling and sampling
            return 500000
        return len(self.data)

    def __getitem__(self, index):
        if not self.train:
            # When testing, pick the selected video (from the precomputed testing set)
            return self.data[index]
        # When training, generate videos on the fly
        x = np.zeros((self.seq_len, self.frame_size, self.frame_size), dtype=np.float32)
        # Generate the trajectories of each digit independently
        for n in range(self.num_digits):
            img = self.data[np.random.randint(len(self.data))]  # Random digit
            trajectory = self._compute_trajectory(*img.shape)  # Generate digit trajectory
            for t in range(self.seq_len):
                sx, sy, _, _ = trajectory[t]
                # Adds the generated digit trajectory to the video
                x[t, sx:sx + img.shape[0], sy:sy + img.shape[1]] += img
        # In case of overlap, brings back video values to [0, 255]
        x[x > 255] = 255
        return x.astype(np.uint8)

    def _compute_trajectory(self, nx, ny, init_cond=None):
        """
        Create a trajectory.

        Parameters
        ----------
        nx : int
            Width of digit image.
        ny : int
            Height of digit image.
        init_cond : tuple
            Optional initial condition for the generated trajectory. It is a tuple of integers (posx, poxy, dx, dy)
            where posx and poxy are the initial coordinates, and dx and dy form the initial speed vector.

        Returns
        -------
        list
            List of tuples (posx, poxy, dx, dy) describing the evolution of the position and speed of the moving
            object. Positions refer to the lower left corner of the object.
        """
        x = []  # Trajectory
        x_max = self.frame_size - nx  # Maximum x coordinate allowed
        y_max = self.frame_size - ny  # Maximum y coordinate allowed
        # Process or create the initial position and speed
        if init_cond is None:
            sx = np.random.randint(0, x_max + 1)
            sy = np.random.randint(0, y_max + 1)
            dx = np.random.randint(-self.max_speed, self.max_speed + 1)
            dy = np.random.randint(-self.max_speed, self.max_speed + 1)
        else:
            sx, sy, dx, dy = init_cond
        # Create the trajectory
        for t in range(self.seq_len):
            # After the movement of a timestep is applied, update the position and speed to take into account
            # collisions with frame borders
            sx, sy, dx, dy = self._process_collision(sx, sy, dx, dy, x_min=0, x_max=x_max, y_min=0, y_max=y_max)
            # Add rounded position and speed to the trajectory
            x.append([int(round(sx)), int(round(sy)), dx, dy])
            # Keep computing the trajectory with exact positions
            sy += dy
            sx += dx
        return x

    def _process_collision(self, sx, sy, dx, dy, x_min, x_max, y_min, y_max):
        """
        Takes as input current object coordinate and speed that might be over the frame borders after the movement of
        the last timestep, and updates them to take into account the object collision with frame borders.

        Parameters
        ----------
        sx : float
            Current object x coordinate, prior to checking whether it collided with a frame border.
        sy : float
            Current object y coordinate, prior to checking whether it collided with a frame border.
        dx : int
            Current object x speed, prior to checking whether it collided with a frame border.
        dy : int
            Current object y speed, prior to checking whether it collided with a frame border.
        x_min : int
            Minimum x coordinate allowed.
        x_max : int
            Maximum x coordinate allowed.
        y_min : int
            Minimum y coordinate allowed.
        y_max : int
            Maximum y coordinate allowed.
        """
        # Check collision on all four edges
        left_edge = (sx < x_min - self.eps)
        upper_edge = (sy < y_min - self.eps)
        right_edge = (sx > x_max + self.eps)
        bottom_edge = (sy > y_max + self.eps)
        # Continue processing as long as a collision is detected
        while (left_edge or right_edge or upper_edge or bottom_edge):
            # Retroactively compute the collision coordinates, using the current out-of-frame position and speed
            # These coordinates are stored in cx and cy
            if dx == 0:  # x is onstant
                cx, cy = (sx, y_min) if upper_edge else (sx, y_max)
            elif dy == 0:  # y is constant
                cx, cy = (x_min, sy) if left_edge else (x_max, sy)
            else:
                a = dy / dx
                b = sy - a * sx
                # Searches for the first intersection with frame borders
                if left_edge:
                    left_edge, n = self._get_intersection_x(a, b, x_min, (y_min, y_max))
                    if left_edge:
                        cx, cy = n
                if right_edge:
                    right_edge, n = self._get_intersection_x(a, b, x_max, (y_min, y_max))
                    if right_edge:
                        cx, cy = n
                if upper_edge:
                    upper_edge, n = self._get_intersection_y(a, b, y_min, (x_min, x_max))
                    if upper_edge:
                        cx, cy = n
                if bottom_edge:
                    bottom_edge, n = self._get_intersection_y(a, b, y_max, (x_min, x_max))
                    if bottom_edge:
                        cx, cy = n
            # Displacement coefficient to get new coordinates after the bounce, taking into account the time left
            # (after all previous displacements) in the timestep to move the object
            p = ((sx - cx) / dx) if (dx != 0) else ((sy - cy) / dy)
            # In the stochastic case, randomly choose a new speed vector
            if not self.deterministic:
                dx = np.random.randint(-self.max_speed, self.max_speed + 1)
                dy = np.random.randint(-self.max_speed, self.max_speed + 1)
            # Reverse speed vector elements depending on the detected collision
            if left_edge:
                dx = abs(dx)
            if right_edge:
                dx = -abs(dx)
            if upper_edge:
                dy = abs(dy)
            if bottom_edge:
                dy = -abs(dy)
            # Compute the remaining displacement to be done during the timestep after the bounce
            sx = cx + dx * p
            sy = cy + dy * p
            # Check again collisions
            left_edge = (sx < x_min - self.eps)
            upper_edge = (sy < y_min - self.eps)
            right_edge = (sx > x_max + self.eps)
            bottom_edge = (sy > y_max + self.eps)
        # Return updated speed and coordinates
        return sx, sy, dx, dy

    def _get_intersection_x(self, a, b, x_lim, by):
        """
        Computes the intersection point of trajectory with the upper or lower border of the frame.

        Parameters
        ----------
        a : float
            dy / dx.
        b : float
            sy - a * sx.
        x_lim : int
            x coordinate of the border of the frame to test the intersection with.
        by : tuple
            Tuple of integers representing the frame limits on the y coordinate.

        Returns
        -------
        bool
            Whether the intersection point lies within the frame limits.
        tuple
            Couple of float coordinates representing the intersection point.
        """
        y_inter = a * x_lim + b
        if (y_inter >= by[0] - self.eps) and (y_inter <= by[1] + self.eps):
            return True, (x_lim, y_inter)
        return False, (x_lim, y_inter)

    def _get_intersection_y(self, a, b, y_lim, bx):
        """
        Computes the intersection point of trajectory with the left or right border of the frame.

        Parameters
        ----------
        a : float
            dy / dx.
        b : float
            sy - a * sx.
        y_lim : int
            y coordinate of the border of the frame to test the intersection with.
        bx : tuple
            Tuple of integers representing the frame limits on the x coordinate.

        Returns
        -------
        bool
            Whether the intersection point lies within the frame limits.
        tuple
            Couple of float coordinates representing the intersection point.
        """
        x_inter = (y_lim - b) / a
        if (x_inter >= bx[0] - self.eps) and (x_inter <= bx[1] + self.eps):
            return True, (x_inter, y_lim)
        return False, (x_inter, y_lim)

    @classmethod
    def make_dataset(cls, data_dir, nx, seq_len, max_speed, deterministic, num_digits, train):
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
        max_speed : int
            Maximum speed of moving digits in the videos.
        deterministic : bool
            Whether to use the deterministic version of the dataset rather that the stochastic one.
        num_digits : int
            Number of digits in each video.
        train : bool
            Whether to use the training or testing dataset.
        """
        if train:
            # When training, only register training MNIST digits
            digits = datasets.MNIST(data_dir, train=train, download=True)
            data = [np.array(img, dtype=np.uint8) for i, (img, label) in enumerate(digits)]
        else:
            # When testining, loads the precomputed videos
            prefix = '' if deterministic else 's'
            dataset = np.load(os.path.join(data_dir, f'{prefix}mmnist_test_{num_digits}digits_{nx}.npz'),
                              allow_pickle=True)
            sequences = dataset['sequences_1']
            data = [sequences[:, i] for i in range(sequences.shape[1])]
        # Create and return the dataset
        return cls(data, nx, seq_len, max_speed, deterministic, num_digits, train)

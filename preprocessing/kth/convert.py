# Converted from https://github.com/edenton/svg/blob/master/data/convert_kth.lua

import argparse
import os
import subprocess


# KTH action classes
classes = ('boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('''
        Generates training and testing videos for the KTH dataset from the original videos,
        and stores them in folder \'processed_${SIZE}\' in the same directory.
        ''')
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where videos from the original dataset are stored.')
    parser.add_argument('--image_size', type=int, metavar='SIZE', default=64,
                        help='Width and height of resulting processed videos.')
    parser.add_argument('--frame_rate', type=int, metavar='RATE', default=25,
                        help='Frame rate at which videos are processed.')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f'Error with data directory: {args.data_dir}')

    # Process videos by class
    for c in classes:
        print(' ---- ')
        print(c)
        c_path = os.path.join(args.data_dir, 'raw', c)
        # Process all videos of a class
        for vid in os.listdir(c_path):
            print(vid)
            if os.path.splitext(vid)[1] != '.avi':
                continue
            fname = vid[:-11]
            print(fname)
            os.makedirs(os.path.join(args.data_dir, f'processed_{args.image_size}', c, fname))
            # Process selected video
            cmd = [
                'ffmpeg',
                '-i', os.path.join(args.data_dir, 'raw', c, vid),
                '-r', str(args.frame_rate),
                '-f', 'image2',
                '-s', f'{args.image_size}x{args.image_size}',
                os.path.join(args.data_dir, f'processed_{args.image_size}', c, fname, f'image-%03d.png')
            ]
            subprocess.call(cmd)

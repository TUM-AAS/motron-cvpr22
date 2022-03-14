import os
import argparse
import tarfile
from io import BytesIO

import numpy as np
import torch
import zarr
from tqdm import tqdm

import sys
sys.path.append('./Motion/')

from motion.quaternion import Quaternion


def process_data(path, out, target_fps=20):
    z_poses = zarr.open(os.path.join(out, 'poses.zarr'), mode='w', shape=(0, 22, 4), chunks=(1000, 22, 4), dtype=np.float32)
    z_trans = zarr.open(os.path.join(out, 'trans.zarr'), mode='w', shape=(0, 3), chunks=(1000, 3), dtype=np.float32)
    z_index = zarr.open(os.path.join(out, 'poses_index.zarr'), mode='w', shape=(0, 2), chunks=(1000, 2), dtype=int)
    i = 0
    tar = tarfile.open(path, 'r')
    for member in tqdm(tar):
        file_name = os.path.basename(member.name)
        if file_name.endswith('.npz') and not file_name.startswith('.'):
            try:
                with tar.extractfile(member) as f:
                    array_file = BytesIO()
                    array_file.write(f.read())
                    array_file.seek(0)
                    data = np.load(array_file)

                    frame_rate = data['mocap_framerate']

                    if not frame_rate % target_fps == 0.:
                        print(f"Warning: FPS does not match for dataset {path}")
                    frame_multiplier = int(np.round(frame_rate / target_fps))

                    body_pose = data['poses'][::frame_multiplier, 0:66].reshape(-1, 22, 3)

                    body_trans = data['trans'][::frame_multiplier]

                    t_body_pose = torch.tensor(body_pose)

                    q_body_pose = Quaternion.qfix_(Quaternion(angle=t_body_pose.norm(dim=-1), axis=t_body_pose).q).numpy()

                    z_poses.append(q_body_pose, axis=0)
                    z_trans.append(body_trans, axis=0)
                    z_index.append(np.array([[i, i + q_body_pose.shape[0]]]), axis=0)
                    i = i + q_body_pose.shape[0]
            except Exception as e:
                print(e)



parser = argparse.ArgumentParser(description='AMASS Process Raw Data')

parser.add_argument('path',
                    type=str,
                    help='Path of the tar files')

parser.add_argument('out',
                    type=str,
                    help='The output path')

parser.add_argument('fps',
                    type=int,
                    default=20,
                    help='FPS')

parser.add_argument('--datasets',
                    type=str,
                    nargs="+",
                    help='The names of the datasets to process',
                    default=None)

args = parser.parse_args()

if __name__ == '__main__':
    in_path = args.path
    out_path = args.out
    fps = args.fps
    datasets = args.datasets
    for dataset in datasets:
        print(f"Processing {dataset}...")
        process_data(os.path.join(in_path, dataset + '.tar.bz2'), os.path.join(out_path, dataset), target_fps=fps)


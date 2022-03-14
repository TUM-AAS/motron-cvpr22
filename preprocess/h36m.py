import argparse
import csv
import os
import re
import zipfile
from glob import glob
from shutil import rmtree

import numpy as np
import torch
from tqdm import tqdm

from motion import Quaternion
from motion.utils.os import maybe_makedir, download_file


def download_convert_h36m_data(dataset_url: str,
                               output_directory: str,
                               output_filename: str) -> None:
    """
    Downloads the h36m dataset in exponential map format and converts it to quaternions.
    """
    maybe_makedir(output_directory)

    output_file_path = os.path.join(output_directory, output_filename)
    if os.path.exists(output_file_path + '.npz'):
        print('The dataset already exists at', output_file_path + '.npz')
    else:
        # Download Human3.6M dataset in exponential map format
        print('Downloading Human3.6M dataset (it may take a while)...')
        h36m_path = output_directory + '/h3.6m.zip'
        download_file(dataset_url, h36m_path, verbose=True)
        print('Extracting Human3.6M dataset...')
        with zipfile.ZipFile(h36m_path, 'r') as archive:
            archive.extractall(output_directory)
        os.remove(h36m_path)  # Clean up

        convert_h36_to_quat(output_directory, output_file_path)


def convert_h36_to_quat(output_directory: str, output_file_path: str) -> None:
    """
    Converts the h36m dataset to quaternions of each joint.
    """
    out_pos = []
    out_rot = []
    out_subjects = []
    out_actions = []

    print('Converting dataset...')
    subjects = sorted(glob(output_directory + '/h3.6m/dataset/*'))
    for subject in tqdm(subjects, unit='Subject'):
        actions = sorted(glob(subject + '/*'))
        for action_filename in actions:
            data = read_file(action_filename)

            # Discard the first joint, which represents a corrupted translation
            data = data[..., 1:, :]
            data = -torch.tensor(data) # Data has to be negated. See: https://github.com/facebookresearch/QuaterNet/issues/10

            q = Quaternion(axis=data, angle=data.norm(dim=-1)).q
            data = Quaternion.qfix_(q).numpy()

            out_pos.append(np.zeros((data.shape[0], 3)))  # No trajectory for H3.6M
            out_rot.append(data)
            tokens = re.split('[/|.]', action_filename.replace('\\', '/'))
            subject_name = tokens[-3]
            out_subjects.append(subject_name)
            action_name = tokens[-2]
            out_actions.append(action_name)

    print('Saving...')
    np.savez_compressed(output_file_path,
                        trajectories=np.asanyarray(out_pos, dtype=object),
                        rotations=np.asanyarray(out_rot, dtype=object),
                        subjects=np.asanyarray(out_subjects, dtype=object),
                        actions=np.asanyarray(out_actions, dtype=object))

    rmtree(output_directory + '/h3.6m')  # Clean up
    print('Done.')


def read_file(path: str) -> np.array:
    """
    Read an individual file in expmap format,
    and return a NumPy tensor with shape (sequence length, number of joints, 3).
    """
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    data = np.array(data, dtype='float64')
    return data.reshape((data.shape[0], -1, 3))


parser = argparse.ArgumentParser(description='H3.6M Download')
parser.add_argument('--dataset_url',
                    type=str,
                    help='The download URL of the dataset in exponential map format',
                    default='http://www.cs.stanford.edu/people/ashesh/h3.6m.zip')
parser.add_argument('--output_directory',
                    type=str,
                    help='The directory where the data should be placed',
                    default='./data/processed/')
parser.add_argument('--output_filename',
                    type=str,
                    help='The name of the npz file which holds the data.',
                    default='h3.6m')

args = parser.parse_args()

if __name__ == '__main__':
    download_convert_h36m_data(args.dataset_url,
                               args.output_directory,
                               args.output_filename)
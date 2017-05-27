import argparse
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd

from load import load_nifti


def update_label(df, noise):

    for label_path, label_warped_path, subject in zip(df["label"], df["label_warped"], df["subject"]):
        if label_warped_path == label_warped_path:
            proba = load_nifti(
                os.path.join(
                    os.path.dirname(label_path),
                    subject + "_segTRI_proba.nii.gz"
                )
            )
            noisy_label, affine = load_nifti(label_warped_path, with_affine=True)
            proba *= noise[noisy_label]
            proba /= np.sum(proba, axis=-1, keepdims=True)
            nib.save(nib.Nifti1Image(proba.astype(np.float32), affine), label_path)


def update_noise(df, n_classes):
    numerator = np.zeros((n_classes, n_classes))
    denominator = np.zeros_like(numerator)

    for label_path, label_warped_path in zip(df["label"], df["label_warped"]):
        print(label_path)
        if label_warped_path == label_warped_path:
            label_warped = load_nifti(label_warped_path)
            proba = load_nifti(label_path)

            for j in range(n_classes):
                indices = np.where(label_warped == j)
                for i in range(n_classes):
                    numerator[j, i] += np.sum(proba[indices, i])
                    denominator[j, i] += np.sum(proba[:, :, :, i])

    noise_matrix = numerator / denominator
    return noise_matrix


parser = argparse.ArgumentParser(description="train VoxResNet with EM alg.")
parser.add_argument(
    "--iteration", "-i", default=10000, type=int,
    help="number of iterations, default=10000")
parser.add_argument(
    "--display_step", "-s", default=1000, type=int,
    help="number of steps to display, default=1000")
parser.add_argument(
    "--gpu", "-g", default=-1, type=int,
    help="negative value indicates no gpu, default=-1")
parser.add_argument(
    "--input_file", "-f", type=str, default="dataset.json",
    help="json file of traininig dataset, default=dataset.json")
parser.add_argument(
    "--n_batch", type=int, default=1,
    help="batch size, default=1")
parser.add_argument(
    "--shape", type=int, nargs='*', action="store",
    default=[80, 80, 80],
    help="shape of input for the network, default=[80, 80, 80]")
parser.add_argument(
    '--out', '-o', default='vrn.npz', type=str,
    help='parameters of trained model, default=vrn.npz')
parser.add_argument(
    "--learning_rate", "-r", default=1e-3, type=float,
    help="update rate, default=1e-3")
parser.add_argument(
    "--weight_decay", "-w", default=0.0005, type=float,
    help="coefficient of l2norm weight penalty, default=0.0005")
parser.add_argument(
    "--em_step", default=5, type=int,
    help="number of EM steps, default=5"
)
args = parser.parse_args()
# print(args)

with open(args.input_file) as f:
    dataset = json.load(f)
train_df = pd.DataFrame(dataset["data"])
n_classes = dataset["n_classes"]

cmd_train_network = (
    "python train.py "
    "-i {} -s {} -g {} -f {} --n_batch {} --shape {} {} {} -r {} -w {} "
    .format(
        args.iteration,
        args.display_step,
        args.gpu,
        args.input_file,
        args.n_batch,
        args.shape[0],
        args.shape[1],
        args.shape[2],
        args.learning_rate,
        args.weight_decay
    )
)

# --output_suffix -o and --model -m are missing
cmd_segment_proba = (
    "python segment_proba.py "
    "-i {} --shape {} {} {} -g {} "
    .format(args.input_file, args.shape[0], args.shape[1], args.shape[2], args.gpu)
)

# noise_matix[j, i] = p(noisy_label == j | true_label == i)
p = 0.1
noise_matrix = (1 - p) * np.eye(n_classes) + p * np.ones((n_classes, n_classes)) / n_classes

# os.system(cmd_train_network + "-o weight_0.npz")

for i in range(1, args.em_step + 1):
    print("=" * 80)
    print("EM step {0:02d}".format(i))

    # E step
    # os.system(cmd_segment_proba + "-m weight_{}.npz".format(i - 1))
    update_label(train_df, noise_matrix)

    # M step
    noise_matrix = update_noise(train_df, n_classes)
    os.system(cmd_train_network + "-o weight_{}.npz".format(i))

os.system("mv weight_{}.npz {}".format(args.em_step, args.out))

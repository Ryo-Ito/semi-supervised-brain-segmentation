import argparse
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd

from load import load_nifti


def copy(df, index):
    for label_path, label_warped_path in zip(df["label"], df["label_warped"]):
        root, ext = os.path.splitext(label_path)
        if ext in [".gz", ".bz2"]:
            root, tmp = os.path.splitext(root)
            ext = tmp + ext
        output = root + "_{}".format(index) + ext
        os.system("cp {} {}".format(label_path, output))
        if label_warped_path == label_warped_path:
            root, ext = os.path.splitext(label_warped_path)
            if ext in [".gz", ".bz2"]:
                root, tmp = os.path.splitext(root)
                ext = tmp + ext
            output = root + "_{}".format(index) + ext
            os.system("cp {} {}".format(label_warped_path, output))


def update_label(df, n_classes, suffix, precision):

    for label_path, label_warped_path, subject in zip(df["label"], df["label_warped"], df["subject"]):
        if label_warped_path == label_warped_path:
            proba = load_nifti(
                os.path.join(
                    os.path.dirname(label_path),
                    subject + suffix
                )
            )
            noisy_label, affine = load_nifti(label_warped_path, with_affine=True)
            assert noisy_label.shape == proba.shape
            klabel = []
            for k in range(n_classes):
                tmp = np.zeros_like(proba)
                tmp[:, :, :, k] = 1
                klabel.append(tmp)
            klabel = np.asarray(klabel)
            assert klabel.shape == (n_classes,) + proba.shape
            likelihood = np.sum(np.exp(-0.5 * precision * (noisy_label - klabel) ** 2), axis=0) * np.power(0.5 * precision / np.pi, 0.5 * n_classes)
            assert likelihood.shape == proba.shape
            posterior = likelihood * proba
            posterior /= np.sum(posterior, axis=-1, keepdims=True)

            nib.save(nib.Nifti1Image(posterior.astype(np.float32), affine), label_path)


def split(inputfile, *outputfile):
    img = nib.load(inputfile)
    data = img.get_data()
    data = data.transpose(3, 0, 1, 2)
    for d, out in zip(data[1:], outputfile):
        nib.save(nib.Nifti1Image(d, img.affine), out)


def merge(outputfile, *inputfile):
    data = []
    for infile in inputfile:
        img = nib.load(infile)
        data.append(img.get_data())
    backgroud = 1 - np.sum(data, axis=0)
    assert (backgroud >= 0.).all() and (backgroud <= 1.).all()
    data.insert(0, backgroud)
    data = np.asarray(data)
    data = data.transpose(1, 2, 3, 0)
    nib.save(nib.Nifti1Image(data, img.affine), outputfile)


def update_displacement(df):

    for label_path, label_warped_path in zip(df["label"], df["label_warped"]):
        if label_warped_path == label_warped_path:
            fixed = ["f{}.nii.gz".format(i) for i in range(1, 4)]
            moving = ["m{}.nii.gz".format(i) for i in range(1, 4)]
            split(label_path, *fixed)
            split(label_warped_path, *moving)
            cmd = (
                "ANTS 3 -i 50x20x10 -t SyN[0.3] -r Gauss[3,0.5] -o tmp "
                "-m MSQ[{0[0]}, {1[0]}, 1, 2] -m MSQ[{0[1]}, {1[1]}, 1, 2] -m MSQ[{0[2]}, {1[2]}, 1, 2]"
                .format(fixed, moving)
            )
            os.system(cmd)
            for m in moving:
                cmd = (
                    "antsApplyTransforms -d 3 -i {} -r {} -o {} -t tmpAffine.txt tmpWarp.nii.gz -n NearestNeighbor"
                    .format(m, label_warped_path, m)
                )
                os.system(cmd)
            merge(label_warped_path, *moving)

    os.system("rm tmp*")
    os.system("rm *.nii.gz")


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
parser.add_argument(
    "--precision", default=10., type=float,
    help="precision parameter of gaussian likelihood function, default=10.")
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

root, ext = os.path.splitext(args.out)

output_file = root + "_0" + ext
os.system(cmd_train_network + "-o {}".format(output_file))

for i in range(1, args.em_step + 1):
    print("=" * 80)
    print("EM step {0:02d}".format(i))

    copy(train_df, i - 1)

    # E step
    suffix = "_segTRI_proba_{}.nii.gz".format(i - 1)
    os.system(
        cmd_segment_proba + "-m {0} -o {1}".format(output_file, suffix)
    )
    update_label(train_df, n_classes, suffix, args.precision)

    # M step
    update_displacement(train_df)
    output_file = root + "_{}".format(i) + ext
    os.system(cmd_train_network + "-o {}".format(output_file))

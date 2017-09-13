import argparse
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i")
    parser.add_argument(
        "--type", "-t", type=str,
        choices=["int32", "float32", "onehot", "argmax"]
    )
    parser.add_argument(
        "--n_classes", "-n", type=int, default=None,
        help="number of classes for onehot conversion"
    )
    args = parser.parse_args()

    img = nib.load(args.input)
    if args.type == "int32":
        data = img.get_data().astype(np.int32)
    elif args.type == "float32":
        data = img.get_data().astype(np.float32)
    elif args.type == "onehot":
        data = img.get_data().astype(np.int)
        data = np.eye(args.n_classes)[data]
        data = data.astype(np.float32)
    elif args.type == "argmax":
        data = img.get_data()
        data = np.argmax(data, axis=-1)
        data = data.astype(np.int32)

    nib.save(nib.Nifti1Image(data, img.affine), args.input)


if __name__ == '__main__':
    main()

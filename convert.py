import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


def mask2proba(arr, sampling=None):
    distance = np.zeros_like(arr)
    for i in range(arr.shape[-1]):
        a = arr[..., i]
        d = distance_transform_edt(1 - a, sampling)
        d[np.nonzero(d)] -= 1
        d = distance_transform_edt(a, sampling) - d
        distance[..., i] = d
    proba = distance - np.max(distance, -1, keepdims=True)
    np.exp(proba, out=proba)
    proba /= proba.sum(-1, keepdims=True)
    return proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i")
    parser.add_argument(
        "--type", "-t", type=str,
        choices=["int32", "float32", "onehot", "argmax", "proba"]
    )
    parser.add_argument(
        "--n_classes", "-n", type=int, default=None,
        help="number of classes"
    )
    parser.add_argument(
        "--coef", "-c", type=float, default=1,
        help="coefficient of distance transform"
    )
    args = parser.parse_args()
    print(args)

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
    elif args.type == "proba":
        data = img.get_data().astype(np.int)
        data = np.eye(args.n_classes)[data]
        zoom = img.header.get_zooms()[:3]
        data = mask2proba(data, np.array(zoom) * args.coef)
        data = data.astype(np.float32)

    nib.save(nib.Nifti1Image(data, img.affine), args.input)


if __name__ == '__main__':
    main()

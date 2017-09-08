import argparse
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str,
        help="input label map"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="output boundary map"
    )
    args = parser.parse_args()
    print(args)

    img = nib.load(args.input)
    data = img.get_data()
    data = np.eye(np.max(data))[data]
    data = np.gradient(data, axis=(0, 1, 2))
    data = np.sqrt(data[0] ** 2 + data[1] ** 2 + data[2] ** 2)
    data = np.sum(data, axis=-1)
    data = np.float32(data)
    assert data.shape == img.shape
    nib.save(nib.Nifti1Image(data, img.affine), args.output)


if __name__ == '__main__':
    main()

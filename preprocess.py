import argparse
import json
import os

from dipy.align.reslice import reslice
import nibabel as nib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk


def preprocess(inputfile, outputfile, order=0, df=None):
    img = nib.load(inputfile)
    data = img.get_data()
    affine = img.affine
    zoom = img.header.get_zooms()[:3]
    data, affine = reslice(data, affine, zoom, (1., 1., 1.), order)
    data = np.squeeze(data)
    data = np.pad(data, [(0, 256 - len_) for len_ in data.shape], "constant")
    if order == 0:
        data = np.int32(data)
        assert data.ndim == 3, data.ndim
    else:
        data_sub = data - gaussian_filter(data, sigma=1)
        img = sitk.GetImageFromArray(np.copy(data_sub))
        img = sitk.AdaptiveHistogramEqualization(img)
        data_clahe = sitk.GetArrayFromImage(img)[:, :, :, None]
        data = np.concatenate((data_clahe, data[:, :, :, None]), 3)
        data = (data - np.mean(data, (0, 1, 2))) / np.std(data, (0, 1, 2))
        assert data.ndim == 4, data.ndim
        assert np.allclose(np.mean(data, (0, 1, 2)), 0.), np.mean(data, (0, 1, 2))
        assert np.allclose(np.std(data, (0, 1, 2)), 1.), np.std(data, (0, 1, 2))
        data = np.float32(data)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, outputfile)


def main():
    parser = argparse.ArgumentParser(description="preprocess dataset")
    parser.add_argument(
        "--input_directory", "-i", type=str,
        help="directory of original dataset")
    parser.add_argument(
        "--template", "-t", type=str,
        help="template subject whose label will be used")
    parser.add_argument(
        "--subjects", "-s", type=str, nargs="*", action="store",
        help="other subject whose label will not be used")
    parser.add_argument(
        "--image_suffix", type=str, default="_ana_strip.nii.gz",
        help="suffix of images, default=_ana_strip.nii.gz")
    parser.add_argument(
        "--label_suffix", type=str, default="_segTRI_ana.nii.gz",
        help="suffix of labels, default=_segTRI_ana.nii.gz")
    parser.add_argument(
        "--output_directory", "-o", type=str,
        help="directory of preprocessed dataset")
    parser.add_argument(
        "--output_file", "-f", type=str, default="dataset.json",
        help="json file of preprocessed dataset, default=dataset.json")
    parser.add_argument(
        "--n_classes", type=int, default=4,
        help="number of classes to classify")
    args = parser.parse_args()
    print(args)

    dataset = {"in_channels": 2, "n_classes": args.n_classes}
    dataset_list = []

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.template is not None:
        output_folder = os.path.join(args.output_directory, args.template)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filedict = {"subject": args.template}

        filename = args.template + args.image_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["image"] = outputfile
        preprocess(
            os.path.join(args.input_directory, args.template, filename),
            outputfile,
            order=1)

        filename = args.template + args.label_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["label"] = outputfile
        preprocess(
            os.path.join(args.input_directory, args.template, filename),
            outputfile,
            order=0)

        dataset_list.append(filedict)

    for subject in args.subjects:
        output_folder = os.path.join(args.output_directory, subject)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filedict = {"subject": subject}

        filename = subject + args.image_suffix
        outputfile = os.path.join(output_folder, filename)
        filedict["image"] = outputfile
        preprocess(
            os.path.join(args.input_directory, subject, filename),
            outputfile,
            order=1)

        if args.template is not None:
            cmd = "ANTS 3 -m PR[{0}, {1}, 1, 2] -i 50x20x10 -t SyN[0.3] -r Gauss[3,0.5] -o tmp".format(os.path.join(args.input_directory, subject, subject + args.image_suffix), os.path.join(args.input_directory, args.template, args.template + args.image_suffix))
            os.system(cmd)
            cmd = (
                "antsApplyTransforms -d 3 -i {} -r {} -o label_tmp.nii.gz -t tmpAffine.txt tmpWarp.nii.gz -n NearestNeighbor"
                .format(
                    os.path.join(args.input_directory, args.template, args.template + args.label_suffix),
                    os.path.join(args.input_directory, subject, subject + args.image_suffix),
                )
            )
            os.system(cmd)

            filename = subject + args.label_suffix
            outputfile = os.path.join(output_folder, filename)
            filedict["label"] = outputfile
            preprocess(
                "label_tmp.nii.gz",
                outputfile,
                order=0)

        dataset_list.append(filedict)

    dataset["data"] = dataset_list

    with open(args.output_file, "w") as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    main()

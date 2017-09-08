import argparse
import nibabel as nib
import numpy as np
import pandas as pd


def boundarize(dataframe, predict_suffix, n_classes):
    converter = np.eye(n_classes)

    for subject, boundary_path, istemplate in zip(dataframe["subject"],
                                                  dataframe["boundary"],
                                                  dataframe["template"]):
        if istemplate:
            continue

        predict_path = subject + "/" + subject + predict_suffix
        img = nib.load(predict_path)
        data = img.get_data()
        data = converter[data]
        data = np.gradient(data, axis=(0, 1, 2))
        data = np.sqrt(data[0] ** 2 + data[1] ** 2 + data[2] ** 2)
        data = np.sum(data, axis=-1)
        data = np.float32(data)
        assert data.shape == img.shape
        nib.save(nib.Nifti1Image(data, img.affine), boundary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", "-i", type=str,
        help="input json file"
    )
    parser.add_argument(
        "--predict_suffix", "-p", type=str,
        help="suffix of predicted label map"
    )
    args = parser.parse_args()
    print(args)

    with open(args.input_file) as f:
        dataset = json.load(f)
    dataframe = pd.DataFrame(dataset["data"])
    boundarize(dataframe, args.predict_suffix, dataset["n_classes"])

if __name__ == '__main__':
    main()

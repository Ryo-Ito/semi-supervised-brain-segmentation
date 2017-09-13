import argparse
import json
import os
import pandas as pd


def throw_with_qsub(cmd):
    cmd = (
        'echo "source ~/.bashrc; cd {0}; {1}" | qsub -l nodes=1:ppn=6'
        .format(os.getcwd(), cmd)
    )
    os.system(cmd)


def perform_registration(dataframe):

    for fixed_subject, fixed_image, fixed_boundary, output_label, source, istemplate in zip(dataframe["subject"], dataframe["original"], dataframe["boundary"], dataframe["label"], dataframe["source"], dataframe["template"]):
        if istemplate:
            continue

        for moving_subject, moving_image, moving_boundary, moving_label in zip(dataframe["subject"], dataframe["original"], dataframe["boundary"], dataframe["label"]):
            if moving_subject == source:
                break

        cmd = (
            "ANTS 3 -i 50x20x10 -t SyN[0.5] -r Gauss[3,0.5] -o from{}to{} "
            "-m CC[{}, {}, 1, 2] -m MSQ[{}, {}, 1, 2]"
            .format(moving_subject, fixed_subject, fixed_image, moving_image, fixed_boundary, moving_boundary)
        )
        cmd += (
            "; antsApplyTransforms -d 3 -i {0} -r {1} -o {2} -t from{3}to{4}Warp.nii.gz from{3}to{4}Affine.txt -n NearestNeighbor"
            .format(moving_label, fixed_image, output_label, moving_subject, fixed_subject)
        )
        cmd += (
            "; python {}/convert.py -i {} -t int32"
            .format(os.path.abspath(os.path.dirname(__file__)), output_label)
        )
        print(fixed_subject, moving_subject)
        throw_with_qsub(cmd)


def main():
    parser = argparse.ArgumentParser(description="estimate transformation")
    parser.add_argument(
        "--input_file", "-i", type=str, default="dataset_train_semi.json",
        help="input json file")
    args = parser.parse_args()
    print(args)

    with open(args.input_file) as f:
        dataset = json.load(f)
    dataframe = pd.DataFrame(dataset["data"])

    perform_registration(dataframe)

if __name__ == '__main__':
    main()

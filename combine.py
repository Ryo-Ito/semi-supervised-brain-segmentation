import argparse
import json
import os


def throw_with_qsub(cmd):
    cmd = (
        'echo "source ~/.bashrc; cd {0}; {1}" | qsub -l nodes=1:ppn=6'
        .format(os.getcwd(), cmd)
    )
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", "-t", type=str,
        help="labeled dataset"
    )
    parser.add_argument(
        "--dataset", "-d", type=str,
        help="non-labeled dataset"
    )
    parser.add_argument(
        "--output_full", type=str,
        help="dataset for full training set"
    )
    parser.add_argument(
        "--output_semi", type=str,
        help="dataset for semi-supervised training"
    )
    parser.add_argument(
        "--boundary_suffix", type=str,
        help="suffix of non-template's boundary image"
    )
    args = parser.parse_args()
    print(args)

    with open(args.template) as f:
        dataset_template = json.load(f)
    data_template = dataset_template["data"]

    with open(args.dataset) as f:
        dataset = json.load(f)
    data = dataset["data"]

    dataset["data"] = data_template + data
    with open(args.output_full, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

    subject_list = []

    for subject in data:
        for template in data_template:
            label_suffix = template["label"].split(template["subject"])[-1]
            subject_ = {
                # predicted label boundary
                "boundary": os.path.join(
                    subject["subject"],
                    subject["subject"] + args.boundary_suffix
                ),
                # transferred label map from template
                "label": os.path.join(
                    subject["subject"],
                    subject["subject"] + "_from_" + template["subject"] + label_suffix
                ),
                "original": subject["original"],
                "preprocessed": subject["preprocessed"],
                "subject": subject["subject"],
                "weight": float(subject["weight"]) / len(data_template),
                "source": template["subject"],
                "template": 0
            }

            cmd = (
                "ANTS 3 -m CC[{}, {}, 1, 2] -i 50x20x10"
                " -t SyN[0.5] -r Gauss[3, 0.5] -o from{}to{}"
                .format(
                    subject_["original"],
                    template["original"],
                    template["subject"],
                    subject_["subject"]
                )
            )
            cmd += (
                "; antsApplyTransforms -d 3 -i {0} -r {1} -o {2} "
                "-t from{3}to{4}Warp.nii.gz from{3}to{4}Affine.txt"
                " -n NearestNeighbor"
                .format(
                    template["label"],
                    subject_["original"],
                    subject_["label"],
                    template["subject"],
                    subject["subject"]
                )
            )
            cmd += (
                "; python {}/convert.py -i {} -t int32"
                .format(
                    os.path.abspath(os.path.dirname(__file__)),
                    subject_["label"]
                )
            )
            throw_with_qsub(cmd)
            subject_list.append(subject_)

    dataset["data"] = data_template + subject_list

    with open(args.output_semi, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
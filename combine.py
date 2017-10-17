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
        "--output_semi", type=str,
        help="dataset for semi-supervised training"
    )
    parser.add_argument(
        "--target_suffix", type=str,
        help="suffix of target label map"
    )
    parser.add_argument(
        "--proba_suffix", type=str,
        help="suffix of propageted probability mask"
    )
    parser.add_argument(
        "--coef", "-c", type=float, default=1,
        help="coefficient of distance transform"
    )
    parser.add_argument(
        "--throw_job", type=int, default=1,
        help="flag indicating to throw job or not"
    )
    args = parser.parse_args()
    print(args)

    with open(args.template) as f:
        dataset_template = json.load(f)
    data_template = dataset_template["data"]

    with open(args.dataset) as f:
        dataset = json.load(f)
    data = dataset["data"]

    subject_list = []

    for subject in data:
        non_template = {
            "proba": [],
            "label": os.path.join(
                subject["subject"],
                subject["subject"] + args.target_suffix
            ),
            "original": subject["original"],
            "preprocessed": subject["preprocessed"],
            "subject": subject["subject"],
            "weight": subject["weight"],
            "template": 0
        }

        for template in data_template:
            non_template["proba"].append(
                os.path.join(
                    subject["subject"],
                    subject["subject"] + "_from_" + template["subject"] + args.proba_suffix
                )
            )

            cmd = (
                "ANTS 3 -m CC[{}, {}, 1, 2] -i 50x20x10"
                " -t SyN[0.5] -r Gauss[3, 0.5] -o from{}to{}"
                .format(
                    non_template["original"],
                    template["original"],
                    template["subject"],
                    non_template["subject"]
                )
            )
            cmd += (
                "; antsApplyTransforms -d 3 -i {0} -r {1} -o {2} "
                "-t from{3}to{4}Warp.nii.gz from{3}to{4}Affine.txt"
                " -n NearestNeighbor"
                .format(
                    template["label"],
                    non_template["original"],
                    non_template["proba"][-1],
                    template["subject"],
                    subject["subject"]
                )
            )
            cmd += (
                "; python {}/convert.py -i {} -t proba -n {} -c {}"
                .format(
                    os.path.abspath(os.path.dirname(__file__)),
                    non_template["proba"][-1],
                    dataset["n_classes"],
                    args.coef
                )
            )
            if args.throw_job:
                throw_with_qsub(cmd)
        subject_list.append(non_template)

    dataset["data"] = data_template + subject_list

    with open(args.output_semi, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()

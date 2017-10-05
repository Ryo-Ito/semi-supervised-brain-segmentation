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
    parser = argparse.ArgumentParser(description="preprocess dataset")
    parser.add_argument(
        "--dataset", "-d", type=str,
        help="dataset to be processed"
    )
    parser.add_argument(
        "--templates", "-t", type=str, nargs="*", action="store",
        help="subject name to be used as templates"
    )
    parser.add_argument(
        "--output_template", type=str,
        help="dataset for template"
    )
    parser.add_argument(
        "--output_semi", type=str,
        help="dataset for semi training"
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
    args = parser.parse_args()
    print(args)

    with open(args.dataset) as f:
        dataset = json.load(f)
    data = dataset["data"]

    template_list = []
    for subject in data:
        if subject["subject"] in args.templates:
            subject["template"] = 1
            template_list.append(subject)

    dataset["data"] = template_list
    with open(args.output_template, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

    subject_list = []

    for subject in data:
        if subject["subject"] in args.templates:
            continue

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

        for template in template_list:
            non_template["proba"].append(
                os.path.join(
                    subject["subject"],
                    subject["subject"] + "_from_" + template["subject"] + args.proba_suffix
                )
            )
            cmd = (
                "ANTS 3 -m PR[{}, {}, 1, 2] -i 50x20x10"
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
            throw_with_qsub(cmd)
        subject_list.append(non_template)

    dataset["data"] = template_list + subject_list

    with open(args.output_semi, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()

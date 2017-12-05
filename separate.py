import argparse
import json
import os


def throw_with_qsub(cmd):
    cmd = (
        'echo "source ~/.bashrc; cd {0}; {1}" | qsub -l nodes=1:ppn=6'
        .format(os.getcwd(), cmd)
    )
    os.system(cmd)


def transfer_label(args):
    with open(args.dataset) as f:
        dataset = json.load(f)
    data = dataset["data"]

    template_list = []
    for subject in data:
        if subject["subject"] in args.templates:
            subject["template"] = 1
            template_list.append(subject)

    n_templates = len(template_list)
    dataset["data"] = template_list
    with open(args.output_template, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

    subject_list = []

    for subject in data:
        label_suffix = subject["label"].split(subject["subject"])[-1]
        if subject["subject"] in args.templates:
            continue

        for template in template_list:
            non_template = {
                "label": os.path.join(
                    subject["subject"],
                    subject["subject"] + "_from_" + template["subject"] + label_suffix
                ),
                "original": subject["original"],
                "preprocessed": subject["preprocessed"],
                "subject": subject["subject"],
                "weight": float(subject["weight"]) / n_templates,
                "source": template["subject"],
                "template": 0
            }
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
                    non_template["label"],
                    template["subject"],
                    subject["subject"]
                )
            )
            cmd += (
                "; python {}/convert.py -i {} -t int32"
                .format(
                    os.path.abspath(os.path.dirname(__file__)),
                    non_template["label"]
                )
            )
            if args.throw_job:
                throw_with_qsub(cmd)
            else:
                os.system(cmd)
            subject_list.append(non_template)

    dataset["data"] = template_list + subject_list

    with open(args.output_semi, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)


def transfer_label_em(args):
    if args.target_suffix is None:
        raise ValueError("assign target_suffix")
    if args.proba_suffix is None:
        raise ValueError("assign proba_suffix")
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
            if args.throw_job:
                throw_with_qsub(cmd)
            else:
                os.system(cmd)
        subject_list.append(non_template)

    dataset["data"] = template_list + subject_list

    with open(args.output_semi, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)


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
        "--throw_job", type=int, default=1,
        help="flag to indicate throw job or not"
    )
    parser.add_argument(
        "--em", type=int, default=1,
        help="flag to indicate which alg. to use, default=1"
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

    if args.em:
        transfer_label_em(args)
    else:
        transfer_label(args)


if __name__ == '__main__':
    main()

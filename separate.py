import argparse
import json
import os
import nibabel as nib
import numpy as np


def onehot_encode(inputfile, *outputfile):
    img = nib.load(inputfile)
    data = img.get_data()
    labels = np.unique(data)
    for label, out in zip(labels, outputfile):
        nib.save(nib.Nifti1Image((data == label).astype(np.float32), img.affine), out)


def split(inputfile, *outputfile):
    img = nib.load(inputfile)
    data = img.get_data().transpose(3, 0, 1, 2)
    for out, d in zip(outputfile, data):
        nib.save(nib.Nifti1Image(d.astype(np.float32), img.affine), out)


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
        "--boundary_suffix", type=str,
        help="suffix of non-template's boundary"
    )
    parser.add_argument(
        "--label_suffix", type=str,
        help="suffix of non-template's target label"
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

    n_templates = len(template_list)
    dataset["data"] = template_list
    with open(args.output_template, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

    subject_list = []

    for subject in data:
        warped_label_suffix = subject["label"].split(subject["subject"])[-1]
        if subject["subject"] in args.templates:
            continue

        for template in template_list:
            non_template = {
                "boundary": os.path.join(
                    subject["subject"],
                    subject["subject"] + args.boundary_suffix
                ),
                "label": os.path.join(
                    subject["subject"],
                    subject["subject"] + "_from_" + template["subject"] + args.label_suffix
                ),
                "warped_label": os.path.join(
                    subject["subject"],
                    subject["subject"] + "_from_" + template["subject"] + warped_label_suffix
                ),
                "original": subject["original"],
                "preprocessed": subject["preprocessed"],
                "subject": subject["subject"],
                "weight": float(subject["weight"]) / n_templates,
                "source": template["subject"]
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
                    non_template["warped_label"],
                    template["subject"],
                    subject["subject"]
                )
            )
            cmd += (
                "; python {}/convert.py -i {} -t int32"
                .format(os.path.abspath(os.path.dirname(__file__)), non_template["warped_label"])
            )
            throw_with_qsub(cmd)
            subject_list.append(non_template)

    dataset["data"] = template_list + subject_list

    with open(args.output_semi, "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()

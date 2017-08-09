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


def throw_with_qsub(cmd, directory):
    cmd = 'echo "source ~/.bashrc; cd {0}; {1}" | qsub -l nodes=1:ppn=6'.format(directory, cmd)
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
        "--output_full", type=str,
        help="dataset for full training"
    )
    parser.add_argument(
        "--output_semi", type=str,
        help="dataset for semi training"
    )
    args = parser.parse_args()
    args.directory = os.path.dirname(args.dataset)
    print(args)

    with open(args.dataset) as f:
        dataset = json.load(f)
    data = dataset["data"]

    data_full = []
    for subject in data:
        dict_ = {
            "label": subject["label"],
            "subject": subject["subject"],
            "weight": subject["weight"],
        }
        directory = os.path.dirname(subject["image"])
        image_path = os.path.join(
            directory,
            subject["subject"] + "_preprocessed.nii.gz"
        )
        dict_["image"] = image_path
        data_full.append(dict_)
    dataset["data"] = data_full
    with open("dataset_train_full.json", "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

    template_list = []
    for subject in data:
        if subject["subject"] in args.templates:
            template = {
                "raw_image": subject["image"],
                "label": subject["label"],
                "onehot": subject["onehot"],
                "subject": subject["subject"],
                "weight": subject["weight"],
                "template": 1
            }
            directory = os.path.dirname(subject["label"])
            image_path = os.path.join(
                directory,
                subject["subject"] + "_preprocessed.nii.gz"
            )
            template["image"] = image_path
            os.system("cp {} {}".format(subject["image"], image_path))
            split(subject["image"], "tmp.nii.gz", subject["image"])
            onehot_list = [
                os.path.join(
                    directory,
                    subject["subject"] + "_segTRI_onehot_{}.nii.gz".format(i)
                ) for i in range(dataset["n_classes"])
            ]
            onehot_encode(subject["label"], *onehot_list)
            template["onehots"] = onehot_list
            template_list.append(template)

    n_templates = len(template_list)
    dataset["data"] = template_list
    with open("dataset_template.json", "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

    subject_list = []

    for subject in data:
        if not subject["subject"] in args.templates:
            for i, template in enumerate(template_list):
                subject_warped_label = {
                    "raw_image": subject["image"],
                    "subject": subject["subject"],
                    "template": 0,
                    "source": template["subject"],
                    "weight": float(subject["weight"]) / n_templates
                }
                directory = os.path.dirname(subject["label"])
                image_path = os.path.join(
                    directory, subject["subject"] + "_preprocessed.nii.gz"
                )
                subject_warped_label["image"] = image_path

                if i == 0:
                    os.system("cp {} {}".format(subject["image"], image_path))
                    split(subject["image"], "tmp.nii.gz", subject["image"])

                output = os.path.join(
                    directory,
                    template["subject"] + "_to_" + subject["subject"] + "_segTRI_ana_0.nii.gz"
                )
                subject_warped_label["onehot"] = os.path.join(
                    directory,
                    template["subject"] + "_to_" + subject["subject"] + "_segTRI_proba.nii.gz"
                )
                subject_list.append(subject_warped_label)
                cmd = (
                    "ANTS 3 -m PR[{}, {}, 1, 2] -i 50x20x10"
                    " -t SyN[0.3] -r Gauss[3,0.5] -o from{}to{}"
                    .format(
                        subject_warped_label["raw_image"],
                        template["raw_image"],
                        template["subject"],
                        subject["subject"]
                    )
                )
                cmd += (
                    "; antsApplyTransforms -d 3 -i {0} -r {1} -o {2} "
                    "-t from{3}to{4}Warp.nii.gz from{3}to{4}Affine.txt"
                    " -n NearestNeighbor"
                    .format(
                        template["label"],
                        subject_warped_label["raw_image"],
                        output,
                        template["subject"],
                        subject["subject"]
                    )
                )
                cmd += ";python convert.py -i {} -d int32".format(output)
                throw_with_qsub(cmd, args.directory)

    dataset["data"] = template_list + subject_list

    with open("dataset_train_semi.json", "w") as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()

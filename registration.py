import argparse
import json
import os
import nibabel as nib
import pandas as pd


def throw_with_qsub(cmd):
    cmd = (
        'echo "source ~/.bashrc; cd {0}; {1}" | qsub -l nodes=1:ppn=6'
        .format(os.getcwd(), cmd)
    )
    os.system(cmd)


def split(inputfile, *outputfile):
    img = nib.load(inputfile)
    data = img.get_data()
    data = data.transpose(3, 0, 1, 2)
    for d, out in zip(data, outputfile):
        nib.save(nib.Nifti1Image(d, img.affine), out)


def perform_registration(df, input_suffix, output_suffix, n_classes):

    for fixed_subject, fixed_image, source, istemplate in zip(df["subject"], df["raw_image"], df["source"], df["template"]):
        if istemplate:
            continue
        fixed_onehot = ["{}_to_{}_onehot{}.nii.gz".format(source, fixed_subject, i) for i in range(n_classes)]
        split(
            os.path.join(
                os.path.dirname(fixed_label),
                source + "_to_" + fixed_subject + input_suffix
            ),
            *fixed_onehot
        )

    for fixed_subject, fixed_image, fixed_label, moving_subject, istemplate in zip(df["subject"], df["raw_image"], df["label"], df["source"], df["template"]):
        if istemplate:
            continue
        for subject, moving_image, moving_label, moving_onehot in zip(df["subject"], df["raw_image"], df["label"], df["onehots"]):
            if moving_subject == subject:
                break
        fixed_onehot = ["{}_to_{}_onehot{}.nii.gz".format(moving_subject, fixed_subject, i) for i in range(n_classes)]
        cmd = (
            "ANTS 3 -i 50x20x10 -t SyN[0.3] -r Gauss[3,0.5] -o from{4}to{5} "
            "-m MSQ[{0[1]}, {1[1]}, 1, 2] -m MSQ[{0[2]}, {1[2]}, 1, 2] -m MSQ[{0[3]}, {1[3]}, 1, 2] -m PR[{2}, {3}, 1, 2]"
            .format(fixed_onehot, moving_onehot, fixed_image, moving_image, moving_subject, fixed_subject)
        )
        output = "{0}/{1}_to_{0}{2}".format(fixed_subject, moving_subject, output_suffix)
        cmd += (
            "; antsApplyTransforms -d 3 -i {0} -r {1} -o {2} -t from{3}to{4}Warp.nii.gz from{3}to{4}Affine.txt -n NearestNeighbor"
            .format(moving_label, fixed_image, output, moving_subject, fixed_subject)
        )
        cmd += (
            "; python {}/convert.py -i {} -d int32"
            .format(os.path.abspath(os.path.dirname(__file__)), output)
        )
        print(fixed_subject, moving_subject)
        throw_with_qsub(cmd)
        # if (fixed_subject, moving_subject) == ("IBSR_06", "IBSR_18"):
        #     os.system(cmd)



def main():
    parser = argparse.ArgumentParser(description="estimate transformation")
    parser.add_argument(
        "--input_file", "-i", type=str, default="dataset_train_semi.json",
        help="input json file")
    parser.add_argument(
        "--input_suffix", "-s", type=str, default="_segTRI_proba_.nii.gz",
        help="input file suffix")
    parser.add_argument(
        "--output_suffix", "-o", type=str, default="_segTRI_ana_.nii.gz",
        help="output file suffix")
    args = parser.parse_args()
    print(args)

    with open(args.input_file) as f:
        dataset = json.load(f)
    df = pd.DataFrame(dataset["data"])
    n_classes = dataset["n_classes"]

    perform_registration(df, args.input_suffix, args.output_suffix, n_classes)

if __name__ == '__main__':
    main()

import argparse
import json
import os
import nibabel as nib
import numpy as np


def softmax(arr, axis=-1):
    y = arr - np.max(arr, axis=axis, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=axis, keepdims=True)
    return y


def main():
    parser = argparse.ArgumentParser(description="perform E step")
    parser.add_argument(
        "--input_file", "-i", type=str,
        help="input training set for semi-supervised learning"
    )
    parser.add_argument(
        "--proba_suffix", "-p", type=str,
        help="suffix of predicted probability map"
    )
    parser.add_argument(
        "--posterior_suffix", "-s", type=str, default=None,
        help="suffix of posterior distribution map"
    )
    args = parser.parse_args()
    print(args)

    with open(args.input_file) as f:
        dataset = json.load(f)["data"]

    for subject in dataset:

        if subject["template"]:
            continue

        prior_path = os.path.join(
            subject["subject"],
            subject["subject"] + args.proba_suffix
        )
        img = nib.load(prior_path)
        prior = img.get_data()
        log_posterior = np.log(prior)

        for obs in subject["proba"]:
            log_posterior += np.log(nib.load(obs).get_data())

        posterior = softmax(log_posterior)
        posterior = posterior.astype(np.float32)
        argmax = posterior.argmax(axis=-1)
        nib.save(nib.Nifti1Image(argmax, img.affine), subject["label"])
        if args.posterior_suffix is not None:
            nib.save(
                nib.Nifti1Image(posterior, img.affine),
                os.path.join(
                    subject["subject"],
                    subject["subject"] + args.posterior_suffix
                )
            )


if __name__ == '__main__':
    main()

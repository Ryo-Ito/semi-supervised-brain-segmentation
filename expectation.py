import argparse
import json
import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

n = 0
while True:
    if os.path.exists(f"weights_{n}.npz"):
        n += 1
    else:
        break

parser = argparse.ArgumentParser(description="perform Expectation step of EM algorithm")
parser.add_argument(
    "--dataset", "-d", type=str,
    help="training dataset"
)
parser.add_argument(
    "--precision", type=float,
    help="precision of gaussian likelihood function"
)
parser.add_argument(
    "--prior_suffix", type=str,
    help="suffix of prior path"
)
parser.add_argument(
    "--observed_suffix", type=str,
    help="suffix of observed path"
)
parser.add_argument(
    "--output_suffix", type=str,
    help="suffix of output path"
)
args = parser.parse_args()
print(args)

with open(args.dataset) as f:
    dataset = json.load(f)
df = pd.DataFrame(dataset["data"])
n_classes = dataset["n_classes"]

mean = [0. for _ in range(n_classes)]
mean[0] = 1.
normal = multivariate_normal(mean=mean, cov=1 / args.precision)
likelihood_same = normal.pdf(mean)
likelihood_diff = normal.pdf(mean[::-1])

for subject, source, target, istemplate in zip(df["subject"],
                                               df["source"],
                                               df["onehot"],
                                               df["template"]):
    if istemplate:
        continue

    # loading prior data and observation data
    prior_path = "{0}/{0}{1}".format(subject, args.prior_suffix)
    observed_path = "{0}/{1}_to_{0}{2}".format(subject, source, args.observed_suffix)
    prior_img = nib.load(prior_path)
    observed_img = nib.load(observed_path)
    prior_data = prior_img.get_data()
    observed_data = observed_img.get_data()
    assert prior_data.ndim == 4
    assert prior_data.shape[3] == n_classes
    assert observed_data.ndim == 3

    # calculate posterior distribution given prior and observation
    likelihood = np.zeros_like(prior_data) + likelihood_diff
    for i in range(n_classes):
        likelihood[observed_data == i, i] = likelihood_same
    assert np.allclose(likelihood.sum(axis=-1), likelihood_same + likelihood_diff * (n_classes - 1))
    posterior = likelihood * prior_data
    posterior /= np.sum(posterior, axis=-1, keepdims=True)

    # save result
    posterior = posterior.astype(np.float32)
    nib.save(
        nib.Nifti1Image(posterior, prior_img.affine),
        f"{subject}/{source}_to_{subject}{args.output_suffix}"
    )
    nib.save(
        nib.Nifti1Image(posterior, prior_img.affine),
        f"{target}"
    )

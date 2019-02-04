# Baysian AB Testing
___
Author: Simon Thornewill von Essen

Date: 2018-10-17
___

## Overview
To begin doing Baysian A-B testing, we'll need to make use of the bayes theorem:

![bayes](http://quicklatex.com/cache3/3d/ql_de45d34fd22f4e7713426c636dccb93d_l3.png)

In english, The `posterior` (afterward) probability is equal to the `prior` (before) times the liklihood of that event happening.

Note that our prior probability distribution is decided by ourselves and how we think the data will look like. If we have no strong convictions then we can use a generic distribution or create something more sophisticated if we have more experience with what we want our posterior to look like.

There are multiple guides that can be used for bayesian inference:

* [PyData](https://www.youtube.com/watch?v=PSqtcNZDj4A)
* [Medium](https://medium.com/@thibalbo/coding-bayesian-ab-tests-in-python-e89356b3f4bd)
* [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

When doing bayesian AB testing, people tend to use their priors to create a beta-distribution which they then sample from to create a distribution. These distributions can be analysed in the same way that we'd analyse the gaussian pdfs in frequentist AB testing.

## Example Code

```
prior_flash = 1
prior_html = 1

N_samp = 1000000

for dependent_var in df_AB.dependant_var.values.tolist():
    regs_flash = 10444
    regs_html5 = 10344
    exec("dep_flash = df_counts.{}.flash".format(dependent_var))
    exec("dep_html5 = df_counts.{}.html5".format(dependent_var))

    A_samples = np.random.beta(dep_flash+prior_flash, 
                               regs_flash-dep_flash+prior_html, 
                               N_samp)

    B_samples = np.random.beta(dep_flash+prior_flash, 
                               regs_html5-dep_html5+prior_html, 
                               N_samp)

    n_bins = 100

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', edgecolor='k')
    plt.hist(A_samples, bins=n_bins, alpha=0.8)
    plt.hist(B_samples, bins=n_bins, alpha=0.8)
    plt.title("{}".format(dependent_var))
    plt.savefig("plots\\bayesian\\figure_{}.png".format(dependent_var))
    plt.show();

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', edgecolor='k')
    plt.hist(B_samples-A_samples, bins=n_bins, alpha=0.8)
    plt.title("{}".format(dependent_var))
    plt.axvline(x=0)
    plt.show();
```

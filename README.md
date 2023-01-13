## Feature Learning for Nonlinear Dimensionality Reduction toward Maximal Extraction of Hidden Patterns

Content
-----
* fealm: FEALM framework and the exemplifying method for UMAP.
* motivating_examples: Scripts used for Motivating Examples.
* computational_evaluations: Scripts used for Computational Evaluations.
* case_studies: Scripts related to Case Studies.

******

Setup
-----

### Requirements
* Python3 (latest)
* Note: Tested on macOS Ventura and Windows 10.

### Setup

* Install FEALM

  * Download/Clone this repository

  * Move to the downloaded repository, then:

    `python3 -m pip install .`

* You can test with sample.py. To run, sample.py you need to install additonal packages.

    `python3 -m pip install matplotlib sklearn`

### Usage
* Examples can be found in "sample.py" and "case_studies".
* API documentation can be found in "fealm/fealm.py".

* Currently, due to the conflict between UMAP and Pathos libraries, a specific loading order of modules is required (see sample.py).

******

### How to cite
T. Fujiwara, Y.-H. Kuo, A. Ynnerman, and K.-L. Ma, "Feature Learning for Nonlinear Dimensionality Reduction toward Maximal Extraction of Hidden Patterns." arXiv:2206.13891
https://arxiv.org/abs/2206.13891

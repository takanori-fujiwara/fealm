## Feature Learning for Dimensionality Reduction toward Maximal Extraction of Hidden Patterns

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
* Python3 (latest), pymanopt (https://github.com/pymanopt/pymanopt)
* Note: Tested on macOS Big Sur and Winows 10.

### Setup
* Install pymanopt from the github repository (DO NOT use pip3 install pymanopt)

  * Download/Clone the repository from https://github.com/pymanopt/pymanopt

  * Move to the downloaded repository, then:

    `rm pyproject.toml`

    `pip3 install .`

* Install FEALM

  * Download/Clone this repository

  * Move to the downloaded repository, then:

    `pip3 install .`

* You can test with sample.py

    `pip3 install matplotlib sklearn`

### Usage
* Examples can be found in "sample.py" and "case_studies"
* API documentation can be found in "fealm/fealm.py" (WIP)

******

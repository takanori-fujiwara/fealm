## Feature Learning for Dimensionality Reduction toward Maximal Extraction of Hidden Patterns

Content
-----
* fealm: FEALM framework and the exemplifying method for UMAP.
* motivating_examples: Scripts used for Motivating Examples.
* computational_evaluations: Scripts used for Computational Evaluations.
* case_studies: Scripts related to Case Studies.

<!-- * Note: The UI is available in this repository: https://github.com/takanori-fujiwara/fealm-ui -->

******

Setup
-----

### Requirements
* Python3 (latest)
* Note: Tested on macOS Big Sur and Windows 10.

### Setup

* Install FEALM

  * Download/Clone this repository

  * Move to the downloaded repository, then:

    `python3 -m pip install .`

* You can test with sample.py. To run, sample.py you need to install additonal packages.

    `python3 -m pip install matplotlib sklearn`

### Usage
* Examples can be found in "sample.py" and "case_studies"
* API documentation can be found in "fealm/fealm.py" (WIP)

* Currently, due to the conflict between UMAP and Pathos libraries, a specific loading order of modules is required (see sample.py). Also, SnC might not work due to the latest hdbscan library's problem for Python 3.10 environment.

******

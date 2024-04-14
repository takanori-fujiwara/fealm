## Feature Learning for Nonlinear Dimensionality Reduction toward Maximal Extraction of Hidden Patterns

About
-----
* The implementation of FEALM, FEALM-UMAP, and NSD from: Feature Learning for Nonlinear Dimensionality Reduction toward Maximal Extraction of Hidden Patterns. Takanori Fujiwara, Yun-Hsin Kuo, Anders Ynnerman, and Kwan-Liu Ma. In Proc. PacificVis, 2023. [arXiv Preprint](https://arxiv.org/abs/2206.13891))

* Demonstration video: https://takanori-fujiwara.github.io/s/fealm/index.html

* Features
  * Feature learning framework, FEALM, for nonliner dimensionality reduction methods. 
  
  * Exemplifying method of FEALM for UMAP (FEALM-UMAP)

  * Various graph dissimilarity measures, including neighbor-disimilarity measure (NSD)

  * Implementation of "Optimal Sets of Projections of High-Dimensional Data" [Lehmann and Theisel, 2016]

  * Generation code of the 2-spheres and 3-class attribute data, which is designed to illustrate nonlinear dimensionality reduction’s sensitiveness to trivial disturbance to manifolds

******

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
* Python3
  
* Note: Tested on macOS Sonoma, Ubuntu 22.0.4 LTS, and Windows 10.

### Setup
* Install FEALM

  * Download/Clone this repository

  * If using Python3.12 (otherwise, skip this step), need to edit and directly install pymanopt as follows because pymanopt installer only supports Python3.11 or older.

    - Download and move to the latest `pymanopt` repo: https://github.com/pymanopt/pymanopt.
    
    - Edit pyproject.toml

      - Line 2: `requires = ["pip>=22.3.1", "setuptools>=65.6.3"]`
      
      - Line 40: `"scipy>=1.0",`

    - Create "_version.py" in "src/pymanopt/" and write down a following line:

      - `__version__ = "2.2.0"`

    - Install `pymanopt`

      - `pip3 install .`

  * Move to the downloaded `fealm` repository, then:

    `pip3 install .`

* You can test with sample.py. To run, sample.py you need to install matplotlib.

    `pip3 install matplotlib`

### Usage
* Examples can be found in "sample.py" and "case_studies".
* API documentation can be found in "fealm/fealm.py".

* Currently, due to the conflict between UMAP and Pathos libraries, a specific loading order of modules is required (see sample.py).

******

### How to cite
T. Fujiwara, Y.-H. Kuo, A. Ynnerman, and K.-L. Ma, "Feature Learning for Nonlinear Dimensionality Reduction toward Maximal Extraction of Hidden Patterns." In Proc. PacificVis, 2023.
(https://arxiv.org/abs/2206.13891)

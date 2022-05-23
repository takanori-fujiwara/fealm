import sys
import os
from distutils.core import setup

setup(name='fealm',
      version=0.4,
      packages=[''],
      package_dir={'': '.'},
      install_requires=[
          'numpy', 'scipy', 'pandas', 'autograd', 'sklearn', 'pymanopt',
          'scikit-learn-extra', 'umap-learn', 'hdbscan', 'pathos', 'networkx',
          'netrd', 'netlsd', 'igraph', 'louvain', 'matplotlib', 'seaborn'
      ],
      py_modules=[
          'fealm', 'fealm.optimization', 'fealm.optimizer',
          'fealm.opt_set_proj', 'fealm.graph_func', 'fealm.graph_dissim',
          'fealm.plot', 'fealm.fealm'
      ])

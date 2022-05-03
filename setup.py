import sys
import os
from distutils.core import setup

setup(name='fealm',
      version=0.1,
      packages=[''],
      package_dir={'': '.'},
      install_requires=[
          'numpy', 'scipy', 'pandas', 'sklearn', 'autograd', 'umap-learn',
          'hdbscan', 'pathos', 'networkx', 'netrd', 'igraph', 'louvain',
          'matplotlib', 'seaborn'
      ],
      py_modules=[
          'fealm', 'fealm.optimization', 'fealm.solver', 'fealm.opt_set_proj',
          'fealm.graph_func', 'fealm.graph_dist', 'fealm.plot', 'fealm.fealm'
      ])

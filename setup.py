import sys
import os
from distutils.core import setup

setup(name='fealm',
      version=0.3,
      packages=[''],
      package_dir={'': '.'},
      install_requires=[
          'numpy', 'scipy', 'pandas', 'autograd', 'sklearn',
          'scikit-learn-extra', 'umap-learn', 'hdbscan', 'pathos', 'networkx',
          'netrd', 'netlsd', 'igraph', 'louvain', 'matplotlib', 'seaborn'
      ],
      py_modules=[
          'fealm', 'fealm.optimization', 'fealm.solver', 'fealm.opt_set_proj',
          'fealm.graph_func', 'fealm.graph_dissim', 'fealm.plot', 'fealm.fealm'
      ])

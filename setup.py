from setuptools import setup

setup(
    name="fealm",
    version=0.7,
    packages=[""],
    package_dir={"": "."},
    install_requires=[
        "cython",
        "numpy",
        "scipy",
        "pandas",
        "autograd",
        "scikit-learn",
        "scikit-learn-extra",
        "pymanopt",
        "umap-learn",
        "hdbscan",
        "pathos",
        "networkx",
        "netrd",
        "netlsd",
        "matplotlib",
        "seaborn",
        "func-timeout",
    ],
    py_modules=[
        "fealm",
        "fealm.optimization",
        "fealm.optimizer",
        "fealm.opt_set_proj",
        "fealm.graph_func",
        "fealm.graph_dissim",
        "fealm.plot",
        "fealm.fealm",
    ],
)
# Note: 'cython' is for solving dependencies, e.g., in hdbscan and scikit-learn-extra.

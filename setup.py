# Setup Python module
from setuptools import setup, find_packages

modules = ["ALLSorts." + p for p in sorted(find_packages("./ALLSorts"))]

setup(
    name="ALLSorts",
    version="0.4.0",
    description="BALL Subtype Classifier/Investigator.",
    url="https://github.com/breons/ALLSorts",
    author="Breon Schmidt",
    license="MIT",
    packages=["ALLSorts", *modules],
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "joblib==0.15.1",
        "matplotlib==3.2.1",
        "scikit-learn==0.22.1",
        "umap-learn==0.4.4",
        "plotly==4.14.3",
        "kaleido==0.1.0"
    ],
    entry_points={
          "console_scripts": ["ALLSorts=ALLSorts.allsorts:run"]
    }
)

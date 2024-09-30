# setup.py
from setuptools import setup, find_packages

setup(
    name='word_similarity_pkg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'sentence-transformers',
        'tqdm',
        'pandas',
        'matplotlib',
    ],
    description='A package for computing word similarity and visualizing results using boxplots.',

)

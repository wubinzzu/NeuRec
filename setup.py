import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="neurec",
    version="2.0.0",
    author="Jonathan Staniforth",
    author_email="jonathanstaniforth@gmail.com",
    description="An open source neural recommender library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/wubinzzu/NeuRec.git",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy==1.16.4',
        'scipy==1.3.1',
        'pandas==0.17',
        'tensorflow==1.12.3',
        'Cython==0.29.14',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    python_requires='>=3.5'
)

import setuptools

setuptools.setup(
    name="neurec",
    version="1.0.0",
    author="Jonathan Staniforth",
    author_email="jonathanstaniforth@gmail.com",
    description="An open source neural recommender library",
    long_description="NeuRec is a comprehensive and flexible Python library for recommendation models that includes a large range of state-of-the-art neural recommender models. This library aims to solve general and sequential (i.e. next-item) recommendation tasks, using the Tensorflow library to provide 26 models out of the box. NeuRec is open source and available under the MIT license. View the GitHub repository for more information: https://github.com/wubinzzu/NeuRec.git",
    url="https://github.com/wubinzzu/NeuRec.git",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5'
)

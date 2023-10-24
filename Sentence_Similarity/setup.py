import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentence_similarity",
    version='1.0.0',
    author="Susheel",
    author_email="susheelnagesh@gmail.com",
    description="Package to calculate the similarity score of two sentences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Susheel-1999/Sentence_Similarity",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    # install_requires=['transformers==4.25.1',
    #                   'sentence-transformers==2.2.2']
)
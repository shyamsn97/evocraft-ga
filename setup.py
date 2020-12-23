import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="evocraft_ga",
    version="0.1.0",
    url="https://github.com/shyamsn97/evocraft-ga",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair97@gmail.com",

    description="Exploring evolutionary algorithms in Minecraft",
    long_description=read("README.md"),

    packages=find_packages(include=["evocraft_ga", "evocraft_ga.*"]),

    install_requires=[
        "torch",
        "Click>=7.0",
        "attrs",
        "numpy==1.19.2",
        "tqdm",
        "sklearn"
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)

from setuptools import find_packages, setup

NAME = "mobileone_pytorch"
DESCRIPTION = "MobileOne implemented in PyTorch."
URL = "https://github.com/federicopozzi33/MobileOne-PyTorch"
EMAIL = "f.pozzi33@campus.unimib.it"
AUTHOR = "Federico Pozzi"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "0.1.1"
REQUIRED = ["torch"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(include=["mobileone_pytorch"]),
    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,
)

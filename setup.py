import os
from setuptools import find_packages, setup


def read(rel_path: str) -> str:
    """read the content of a file"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding='utf-8') as file:
        return file.read()


def get_version(rel_path: str) -> str:
    """read the version inside the __init__.py file"""
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")

install_requires = [
    "simplejson",
    "pandas",
    "numpy",
    "Pillow",
    "cython",
    "matplotlib",
    "scikit-image",
    "opencv-python",
    "h5py",
    "imgaug",
    "argparse",
    "scikit-learn",
    "slidingwindow",
    "pyyaml"
]
tests_require = [
    "pytest",
]
docs_require = [
    "sphinx"
]
extras_require = {
    "tests": install_requires + tests_require,
    "docs": install_requires + docs_require,
    "dev": install_requires + tests_require + docs_require,
}

setup(
    name="TACTUS - model",
    version=get_version("src/__init__.py"),
    description="Threatening activities classification toward users' security",
    long_description=long_description,
    classifiers=[
        "Development Status :: 5 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github/Cranfield-GDP3/TACTUS-model",
    project_urls={
        "issues": "https://github/Cranfield-GDP3/TACTUS-model/issues",
    },
    python_requires=">=3.9",
    packages=find_packages(where="."),
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
)

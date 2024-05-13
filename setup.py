from setuptools import setup, find_packages
from distutils.util import convert_path

vpath = convert_path("openECCI/version.py")
with open("README.md", "r") as fid:
    long_description = fid.read()

# Get release information without importing anything from the project
with open("openECCI/release.py") as fid:
    for line in fid:
        if line.startswith("author"):
            AUTHOR = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("maintainer_email"):  # Must be before 'maintainer'
            MAINTAINER_EMAIL = line.strip(" = ").split()[-1][1:-1]
        elif line.startswith("maintainer"):
            MAINTAINER = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("name"):
            NAME = line.strip().split()[-1][1:-1]
        elif line.startswith("version"):
            VERSION = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("license"):
            LICENSE = line.strip().split(" = ")[-1][1:-1]
        elif line.startswith("platforms"):
            PLATFORMS = line.strip().split(" = ")[-1][1:-1]

# TODO: Add documentation and tests to the package in the future
# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
# fmt: off
extra_feature_requirements = {
    "tests": ["pytest>=5.0"],
    "coverage": ["pytest-cov", "codecov"],
    "build-doc": [
        "sphinx>=4.3.0",
        "sphinx_rtd_theme>=0.5.1",
        "sphinx-toggleprompt",
    ],
}
# fmt: on

setup(
    # Package description
    name=NAME,
    version=VERSION,
    license=LICENSE,
    url="https://github.com/IMBalENce/openECCI",
    python_requires=">=3.6",
    description=(
        "An open source python package for guiding Electron Channelling Contrast Imaging (ECCI)"
        "in Scanning Electron Microscopes (SEM)."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        (
            "License :: OSI Approved :: GNU General Public License v3 or later "
            "(GPLv3+)"
        ),
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    platforms=PLATFORMS,
    keywords=[
        "ECCI",
        "electron channelling contrast imaging",
        "ECP",
        "electron channeling pattern",
        "EBSD",
        "electron backscatter diffraction",
        "RKD",
        "reflected kikuchi diffraction",
        "SEM",
        "scanning electron microscopy",
        "kikuchi pattern",
        "dislocation",
        "defect analsysis",
    ],
    zip_safe=True,
    # Contact
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    download_url="https://pypi.python.org/pypi/open_ecci",
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    project_urls={
        "Bug Tracker": "https://github.com/IMBalENce/openECCI/issues",
        "Source Code": "https://github.com/IMBalENce/openECCI",
    },
    # Dependencies
    install_requires=[
        "hyperspy           >= 1.7.3",
        "h5py               >= 2.10",
        "matplotlib         >= 3.5",
        "numpy              >= 1.21.6",
        "orix               >= 0.11.1",
        "tqdm               >= 4.58.0",
        "scipy              >= 1.7",
        "kikuchipy          >= 0.8.6",
        "pyqt5-tools >= 5.15",
        "opencv-python     >= 4.5.3",
        "tifffile          >= 2023.2.2",
    ],
    extras_require=extra_feature_requirements,
    # Files to include when distributing package
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["LICENSE.txt", "README.md"], "openECCI": ["*.py", "data/*"]},
)

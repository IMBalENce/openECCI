<div align="center">
  <a href="https://github.com/IMBalENce/openECCI">
    <img width="60%" src="https://github.com/IMBalENce/openECCI/raw/main/images/logo/banner.png">
  </a>
</div>
<br>
OpenECCI [open-e-ki] is a Python library for Electron Channelling Contrast Imaging (ECCI) analysis of crystal defects in Scanning Electron Microscope (SEM) by guiding precise stereo-tilt of the specimen to the preferable orientations. 
<br> 

| Deployment    | ![PyPI - Version](https://img.shields.io/pypi/v/openECCI) | 
| :- | :- |
| **Activity**      | Work in progress |
| **Citation**      | [![DOI](https://zenodo.org/badge/799454158.svg)](https://zenodo.org/doi/10.5281/zenodo.11183729) |
| **License**       | [![GitHub License](https://img.shields.io/github/license/IMBalENce/openECCI)](https://opensource.org/licenses/GPL-3.0) |

## Documentation

Refer to the included [Jupyter notebook tutorials](https://github.com/IMBalENce/openECCI/tree/main/tutorials) for detailed instruction and applications. Demo datasets can be downloaded from [openECCI-data](https://github.com/IMBalENce/openECCI-data) repository. The [openECCI documentation](https://openecci-tutorials.readthedocs.io/en/latest/#) page is currently under development. Stay tuned for more information. More code examples, and a detailed workflow tutorial paper will be updated soon.

## Installation

Create a new conda environment
```
conda update conda
conda create -n openecci python=3.10 
conda activate openecci
``` 
Install openECCI with `pip`:
```bash
pip install openECCI
```

or you can install from source:
```
git clone https://github.com/IMBalENce/openECCI
cd openECCI
pip install .
```
If you want to create an editable install:
```
pip install -e .
```

## Citing openECCI

If you are using openECCI in your scientific research, please help our scientific
visibility by citing the Zenodo DOI: https://doi.org/10.5281/zenodo.11183730.

## Contributors
| | |
| :- | :- |
|Zhou Xu |  Monash Centre for Electron Microscopy (MCEM) |
| Håkon Wiik Ånes | Norwegian University of Science and Technology <br> Xnovo Technology Aps |
| Sergey Gorelick | Monash Centre for Electron Microscopy (MCEM) <br>  Ramaciotti Centre for Cryo-Electron Microscopy |
| Xiya Fang | Monash Centre for Electron Microscopy (MCEM) |
| Peter Miller | Monash Centre for Electron Microscopy (MCEM) |

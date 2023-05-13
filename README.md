<p align="center">
  <img src="doc/static/logo.png" width="450", alt="MadNIS 2">
</p>

<h2 align="center">Neural Multi-Channel Importance Sampling</h2>

<p align="center">
<a href="https://arxiv.org/abs/2212.06172"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2212.06172-b31b1b.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://www.tensorflow.org"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-2.9.2-DD6C3A.svg?style=flat&logo=tensorflow"></a>
<a href="https://pypi.org/project/pip/"><img alt="PyPi license" src="https://badgen.net/pypi/license/pip/"></a>
</p>

This a machine learning framework to perform neural multi-channel importance sampling in MadGraph.
It containes modules to construct a machine-learning based
Monte Carlo integrator using TensorFlow 2.

## Installation

```bash
# clone the repository
git clone https://github.com/madgraph-ml/madnis-tf
# then install in dev mode
cd madnis-tf
python setup.py develop
```

## Citation

If you use this code or parts of it, please cite:

    @article{Heimel:2022wyj,
    author = "Heimel, Theo and Winterhalder, Ramon and Butter, Anja and Isaacson, Joshua and 
    Krause, Claudius and Maltoni, Fabio and Mattelaer, Olivier and Plehn, Tilman",
    title = "{MadNIS -- Neural Multi-Channel Importance Sampling}",
    eprint = "2212.06172",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "IRMP-CP3-22-56, MCNET-22-22, FERMILAB-PUB-22-915-T",
    month = "12",
    year = "2022"}

=================
MadNIS
=================

This a machine learning framework to perform neural multi-channel importance sampling in MadGraph.
It containes modules to construct a machine-learning based
Monte Carlo integrator using TensorFlow 2.


Installation
-------------

.. code:: sh

   # clone the repository
   git clone https://github.com/ramonpeter/MadNIS.git
   # then install in dev mode
   cd MadNIS
   python setup.py develop

Citation
---------

If you use this code or parts of it, please cite:

.. code:: sh

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

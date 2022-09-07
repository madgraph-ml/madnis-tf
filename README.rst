=================
MadNIS
=================

This a machine learning framework to perform neural multi-channel importance sampling in MadGraph.
It containes modules to construct a machine-learning based
Monte Carlo integrator using TensorFlow 2.


Installation
-------------

Dependencies
~~~~~~~~~~~~

+---------------------------+-------------------------------+
| **Package**               | **Version**                   |
+---------------------------+-------------------------------+
| Python                    | >= 3.7                        |
+---------------------------+-------------------------------+
| Tensorflow                | >= 2.7.0                      |
+---------------------------+-------------------------------+
| Numpy                     | >= 1.20.0                     |
+---------------------------+-------------------------------+
| lhereader                 | >= 1.0.11                     |
+---------------------------+-------------------------------+

Download + Install
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   # clone the repository
   git clone https://github.com/ramonpeter/MadNIS.git
   # then install in dev mode
   cd MadNIS
   python setup.py develop

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pdfflow import mkPDFs
import tensorflow as tf
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import lhapdf

class Colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

MZ = 9.110000e+01
X1 = [ np.random.uniform(0,1,(1,)) for i in range(2)]
PID = 1

pdf_pdfflow = mkPDFs("NNPDF31_lo_as_0118", [0])
pdf_lhapdf = lhapdf.mkPDF("NNPDF31_lo_as_0118/0")

for x in X1:
    # PDFFLow
    x1 = tf.constant(x)#tf.random.uniform([10], dtype=tf.float64)
    q2 = tf.ones([1], dtype=tf.float64) * MZ**2
    pid = tf.cast([PID], dtype=tf.int32)

    # pdfs
    pdf_mg5 = pdf_lhapdf.xfxQ2(PID, x[0], MZ**2)
    pdf_madnis = pdf_pdfflow.xfxQ2(pid, x1, q2)

    print("> momentum fraction:")
    print(f"x1 : {x[0]}")

    print("> lhapdf  f(x1, Q2)*x1 : %s%.16e%s"%(Colour.GREEN,pdf_mg5,Colour.END))
    print("> pdfflow f(x1, Q2)*x1 : %s%.16e%s"%(Colour.YELLOW,pdf_madnis,Colour.END))
    print("")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pdfflow import mkPDFs
import tensorflow as tf
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

MZ = 9.110000e+01
samples = 50
pdf = mkPDFs("NNPDF31_lo_as_0118", [0])
x1 = tf.constant(np.logspace(-3.0, 0.0, num=samples))#tf.random.uniform([10], dtype=tf.float64)
q2 = tf.ones([samples], dtype=tf.float64) * 1e4
pid = tf.cast([2], dtype=tf.int32)

with tf.GradientTape() as g:
    g.watch(x1)
    pdf_u = pdf.xfxQ2(tf.cast([2], dtype=tf.int32), x1, q2)
    pdf_ubar = pdf.xfxQ2(tf.cast([-2], dtype=tf.int32), x1, q2)
    pdf_u_val = pdf_u - pdf_ubar

grad_pdf = g.gradient(pdf_u_val, x1)*x1


plt.rc("text", usetex=True)
plt.rc('font', family='serif', size=12.0)
plt.rc('axes', labelsize='large')
fig, ax1 = plt.subplots()
ax1.plot(x1[::-1], pdf_u[::-1], label=r"$\mathrm{u}$")
ax1.plot(x1[::-1], pdf_ubar[::-1], label=r"$\bar{\mathrm{u}}$")
ax1.plot(x1[::-1], pdf_u_val[::-1], label=r"$\mathrm{u}_\mathrm{val}$")
ax1.plot(x1[::-1], grad_pdf[::-1], label=r"$x\frac{\partial\mathrm{u}_\mathrm{val}}{\partial x}$")
ax1.legend(frameon=False)
ax1.set_xscale("log")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$xf(x,Q^2)$")
ax1.axhline(y=0,linewidth=1, linestyle='--', color='grey')
fig.savefig("test_pdfflow.pdf", bbox_inches="tight")
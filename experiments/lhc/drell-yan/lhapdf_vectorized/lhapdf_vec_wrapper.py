import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_void_p, c_char_p, c_double
import os

double_arr = npct.ndpointer(dtype=np.float64, ndim=1, flags="CONTIGUOUS")
pdf_lib = npct.load_library("lhapdf_vectorized", os.path.dirname(os.path.realpath(__file__)))
pdf_lib.lhapdf_vec_init.argtypes = [c_char_p, c_int]
pdf_lib.lhapdf_vec_init.restype = c_void_p
pdf_lib.lhapdf_vec_xfxQ2.argtypes = [c_void_p, c_int, double_arr, c_double, double_arr, c_int]
pdf_lib.lhapdf_vec_xfxQ2.restype = None
pdf_lib.lhapdf_vec_xmax.argtypes = [c_void_p]
pdf_lib.lhapdf_vec_xmax.restype = c_double
pdf_lib.lhapdf_vec_free.argtypes = [c_void_p]
pdf_lib.lhapdf_vec_free.restype = None

class LhaPdfWrapper:
    def __init__(self, name, member):
        self.pdf = pdf_lib.lhapdf_vec_init(name.encode("ascii"), member)

    def xfxQ2(self, pid, x, q2):
        x_flat = x.flatten()
        out = np.empty_like(x_flat)
        pdf_lib.lhapdf_vec_xfxQ2(self.pdf, pid, x_flat, q2, out, len(x_flat))
        return out.reshape(x.shape)

    def xmax(self):
        return pdf_lib.lhapdf_vec_xmax(self.pdf)

    def __del__(self):
        pdf_lib.lhapdf_vec_free(self.pdf)

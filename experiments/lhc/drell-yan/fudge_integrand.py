from typing import List
import math as m
import warnings

import tensorflow as tf
from pdfflow import mkPDFs

warnings.filterwarnings("ignore")

# -------------------------------------
# Basic Inputs
# -------------------------------------
MW = 8.041900e01  # W-Boson Mass   8.041900e01 (MG5)
MZ = 9.118800e01  # Z-Boson Mass   9.118800e01 (MG5)
WZ = 2.441404e-00  # Z-Boson Width  2.441404e00 (MG5)
MZP = 4.000000e02  # Z'-Boson Mass
WZP = 5.000000e-01  # Z'-Boson Width
GF = 1.166390e-05  # Fermi Constant 1.166390e-05 (MG5)
NC = 3  # Color factor
# -------------------------------------


class FudgeDrellYan:
    r"""Amplitudes, weights and differential cross section
    for the Drell-Yan process

        q q~ > Z'/Z/gamma > l+ l-

    where q = {u,c,b,d,s}.
    """

    def __init__(
        self,
        isq: List[str],
        e_had: float = 13000.0,
        mw: float = MW,
        mz: float = MZ,
        wz: float = WZ,
        mzp: float = MZP,
        wzp: float = WZP,
        gf: float = GF,
        pdfset: str = "NNPDF40_lo_as_01180",
        pdflib: str = "lhapdf",
        input_format: str = "cartesian",
        z_scale: float = 1.0,
        zp_scale: float = 1.0,
        **kwargs
    ):
        """
        Args:
            isq (List[str]): List of initial state quarks. Possible are:
                up-quark type   : `"u"`, `"c"`,
                down-quark type : `"d"`, `"s"`, `"b"`
            e_had (float, optional): Center-of-mass energy in LAB frame. Defaults to 13000.0.
            mw (float, optional): W-boson mass. Defaults to MW.
            mz (float, optional): Z-boson mass. Defaults to MZ.
            wz (float, optional): Z-boson width. Defaults to WZ.
            mzp (float, optional): Z'-boson mass. Defaults to MZP.
            wzp (float, optional): Z'-boson width. Defaults to WZP.
            gf (float, optional): Fermi constant. Defaults to GF.
            pdfset (str, optional): PDF-set to use.. Defaults to "NNPDF40_lo_as_01180".
                Requires `lhapdf` and `lhapdf-management`.
            pdflib (str, optional): PDF library to use.
                Options "lhapdf" (default) or "pdfflow"
            input_format (str,optional): Which parametrization do the momenta have.
                Default is `"cartesian"` which means `p = {px1,py1,pz1,pz2}`. Alternative
                is `"convpolar"` as `p = {x1,x2,costheta,phi}`.
        """
        super().__init__(**kwargs)
        self._dtype = tf.keras.backend.floatx()

        # Input parameters
        self.e_had = tf.constant(e_had, dtype=self._dtype)
        self.s_had = tf.constant(e_had**2, dtype=self._dtype)
        self.mw = tf.constant(mw, dtype=self._dtype)
        self.mz = tf.constant(mz, dtype=self._dtype)
        self.wz = tf.constant(wz, dtype=self._dtype)
        self.mzp = tf.constant(mzp, dtype=self._dtype)
        self.wzp = tf.constant(wzp, dtype=self._dtype)
        self.gf = tf.constant(gf, dtype=self._dtype)
        self.z_scale = tf.constant(z_scale, dtype=self._dtype)
        self.zp_scale = tf.constant(zp_scale, dtype=self._dtype)

        # Define input format
        self.input_format = input_format

        # Factorisation scale and pdfset
        self.muf2 = self.mz**2
        if pdflib == "lhapdf":
            from lhapdf_vectorized.lhapdf_vec_wrapper import LhaPdfWrapper
            pdf = LhaPdfWrapper(pdfset, 0)

            def pdffunc(x1, x2, pid):
                q2 = tf.get_static_value(self.muf2)
                pdf_plus = lambda x: pdf.xfxQ2(pid, x, q2) 
                pdf_minus = lambda x: pdf.xfxQ2(-pid, x, q2) 
                pdf_1a = tf.numpy_function(pdf_plus, [x1], self._dtype) / x1
                pdf_2a = tf.numpy_function(pdf_minus, [x2], self._dtype) / x2
                pdf_1b = tf.numpy_function(pdf_minus, [x1], self._dtype) / x1
                pdf_2b = tf.numpy_function(pdf_plus, [x2], self._dtype) / x2
                return pdf_1a * pdf_2a + pdf_1b * pdf_2b
        elif pdflib == "pdfflow":
            from pdfflow import mkPDFs
            pdf = mkPDFs(pdfset, [0])

            def pdffunc(x1, x2, pid):
                pid = tf.cast([pid], dtype=tf.int32)
                q2 = tf.cast(tf.ones_like(x1) * self.muf2, dtype=self._dtype)
                pdf_1a = tf.convert_to_tensor(pdf.xfxQ2(pid, x1, q2)) / x1
                pdf_2a = tf.convert_to_tensor(pdf.xfxQ2(-pid, x2, q2)) / x2
                pdf_1b = tf.convert_to_tensor(pdf.xfxQ2(-pid, x1, q2)) / x1
                pdf_2b = tf.convert_to_tensor(pdf.xfxQ2(pid, x2, q2)) / x2
                return pdf_1a * pdf_2a + pdf_1b * pdf_2b
        else:
            raise ValueError(f"Unknown pdf library '{pdflib}'")
        self._eval_pdfs = pdffunc

        # Basic Definitions
        self.cw2 = self.mw**2 / self.mz**2
        self.sw2 = 1 - self.cw2

        # Z' Definitions
        self.cw2p = self.mw**2 / self.mzp**2
        self.sw2p = 1 - self.cw2p

        # Definition of electroweak coupling
        # self.alpha = m.sqrt(2) * self.gf * self.mw**2 * self.sw2 # Gf scheme
        self.alpha = tf.constant(1 / 1.325070e02, dtype=self._dtype)  # MG-Input

        # Define list of initial state quarks
        self.isq = isq
        assert isinstance(self.isq, list)

        # Define couplings and charges
        self.V_l = -1 / 2 + 2 * self.sw2
        self.A_l = -1 / 2

        self.Q_f = {
            "d": -1 / 3,
            "s": -1 / 3,
            "u": 2 / 3,
            "c": 2 / 3,
            "b": 2 / 3,
        }

        self.V_q = {
            "d": -1 / 2 + 2 / 3 * self.sw2,
            "s": -1 / 2 + 2 / 3 * self.sw2,
            "u": +1 / 2 - 4 / 3 * self.sw2,
            "c": +1 / 2 - 4 / 3 * self.sw2,
            "b": +1 / 2 - 4 / 3 * self.sw2,
        }

        self.A_q = {
            "d": -1 / 2,
            "s": -1 / 2,
            "u": 1 / 2,
            "c": 1 / 2,
            "b": 1 / 2,
        }

    def a_ZZ(self, s: tf.Tensor, isq: str):
        m_ZZ = (
            (self.V_q[isq] ** 2 + self.A_q[isq] ** 2)
            * (self.V_l**2 + self.A_l**2)
            / 4
        )
        factor_e2 = 4 * m.pi * self.alpha
        factor_g2 = factor_e2 / (self.cw2 * self.sw2)
        return s**2 / 4 * factor_g2**2 * m_ZZ

    def a_yy(self, s: tf.Tensor, isq: str):
        m_yy = 4 * self.Q_f[isq] ** 2
        factor_e2 = 4 * m.pi * self.alpha
        return s**2 / 4 * factor_e2**2 * m_yy

    def a_yZ(self, s: tf.Tensor, isq: str):
        m_yz = (-1) * self.Q_f[isq] * self.V_q[isq] * self.V_l
        factor_e2 = 4 * m.pi * self.alpha
        factor_g2 = factor_e2 / (self.cw2 * self.sw2)
        return s**2 / 4 * factor_g2 * factor_e2 * m_yz

    def b_ZZ(self, s: tf.Tensor, isq: str):
        m_ZZ = (-1) * self.A_q[isq] * self.A_l * self.V_q[isq] * self.V_l
        factor_e2 = 4 * m.pi * self.alpha
        factor_g2 = factor_e2 / (self.cw2 * self.sw2)
        return s**2 / 2 * factor_g2**2 * m_ZZ

    def b_yZ(self, s: tf.Tensor, isq: str):
        m_yz = self.Q_f[isq] * self.A_q[isq] * self.A_l
        factor_e2 = 4 * m.pi * self.alpha
        factor_g2 = factor_e2 / (self.cw2 * self.sw2)
        return s**2 / 2 * factor_g2 * factor_e2 * m_yz

    def prop_factor(self, s: tf.Tensor, m1: float, m2: float, w1: float, w2: float):
        nom = s**2 - s * (m1**2 + m2**2) + m1**2 * m2**2 + w1 * w2 * m1 * m2
        den1 = (s - m1**2) ** 2 + w1**2 * m1**2
        den2 = (s - m2**2) ** 2 + w2**2 * m2**2
        return nom / (den1 * den2)

    def amp2_single(self, cos_theta: tf.Tensor, s: tf.Tensor, isq: str):
        """Squared single diagram matrix element for a given production mode

        Args:
            cos_theta (tf.Tensor): scattering angle in partonic CM frame.
            s (tf.Tensor): partonic CM energy.
            mode (str): Which diagram: `"ZZ"`, `"yy"`,`"yZ"`
            isq (str): initial-state quark.

        Returns:
            m2: Squared single diagram matrix element (|M_Z|^2 or |M_y|^2),
                or Re(M_yM^*_Z) for interference depending on the mode.
        """
        n_spins = 2
        Kyy = self.a_yy(s, isq) * (1 + cos_theta**2)
        Kzz = self.a_ZZ(s, isq) * (1 + cos_theta**2) + self.b_ZZ(s, isq) * cos_theta
        Kyz = self.a_yZ(s, isq) * (1 + cos_theta**2) + self.b_yZ(s, isq) * cos_theta

        # Squares
        m_yy = Kyy * n_spins**2 * self.prop_factor(s, 0, 0, 0, 0)
        m_ZZ = (
            Kzz * n_spins**2 * self.prop_factor(s, self.mz, self.mz, self.wz, self.wz)
        )
        m_ZpZp = (
            Kzz
            * n_spins**2
            * self.prop_factor(s, self.mzp, self.mzp, self.wzp, self.wzp)
        )

        # interferences
        m_ZZp = (
            Kzz
            * n_spins**2
            * self.prop_factor(s, self.mz, self.mzp, self.wz, self.wzp)
        )
        m_yZp = Kyz * n_spins**2 * self.prop_factor(s, 0, self.mzp, 0, self.wzp)
        m_yZ = Kyz * n_spins**2 * self.prop_factor(s, 0, self.mz, 0, self.wz)

        return m_yy, m_ZZ, m_ZpZp, m_ZZp, m_yZp, m_yZ

    def amp2_all(self, cos_theta: tf.Tensor, s: tf.Tensor, isq: str):
        """Full squared matrix element.

        Args:
            cos_theta (tf.Tensor): scattering angle in partonic CM frame.
            s (tf.Tensor): partonic CM energy.
            isq (str): initial-state quark.

        Returns:
            m2: Full squared matrix element (|M_Z + M_y|^2)
        """
        # Squares
        m_yy, m_ZZ, m_ZpZp, m_ZZp, m_yZp, m_yZ = self.amp2_single(cos_theta, s, isq)
        m_squares = m_yy + self.z_scale * m_ZZ + self.zp_scale * m_ZpZp
        # interferences
        m_int = (
            tf.math.sqrt(self.z_scale) * m_yZ
            + tf.math.sqrt(self.zp_scale) * m_yZp
            + tf.math.sqrt(self.z_scale * self.zp_scale) * m_ZZp
        )
        return m_squares + 2 * m_int

    def partonic_dxs(self, cos_theta: tf.Tensor, s: tf.Tensor, isq: str):
        """Fully differential partonic cross section, i.e.

            dsigma/dOmega = 1/(64 PI^2) * 1/s * 1/(4 * NC) * |M|^2

        which includes:
            - ps-weight: 1/(32 PI^2)
            - 1/flux = 1/(2s)
            - averaging/summation over initial/final-state spins: 1/4
            - averaging over initial-state color: 1/NC

        Args:
            cos_theta (tf.Tensor): scattering angle in partonic CM frame.
            s (tf.Tensor): partonic CM energy.
            isq (str): initial-state quark.

        Returns:
            p_dxs (tf.Tensor): partonic differential cross-section
        """
        cs_factor = 1 / (4 * NC)
        ps_weight = 1 / (32 * m.pi**2)
        fluxm1 = 1 / (2 * s)
        return fluxm1 * ps_weight * cs_factor * self.amp2_all(cos_theta, s, isq)

    def hadronic_dxs(
        self,
        x1: tf.Tensor,
        x2: tf.Tensor,
        cos_theta: tf.Tensor,
        isq: str,
    ):
        """Fully differential hadronic cross section, including the
        parton density functions (PDFs).

        Args:
            x1 (tf.Tensor): momentum fraction x1.
            x2 (tf.Tensor): momentum fraction x2.
            cos_theta (tf.Tensor): scattering angle in partonic CM frame.
            isq (str): initial-state quark.

        Returns:
            h_dxs (tf.Tensor): hadronic differential cross-section
        """
        # Get Particle ID
        pid = {"d": 1, "u": 2, "s": 3, "c": 4, "b": 5}[isq]

        # Calculate momentum fractions and partonic CM energy
        x1 = tf.cast(x1, dtype=self._dtype)
        x2 = tf.cast(x2, dtype=self._dtype)
        s_parton = x1 * x2 * self.s_had

        # Calculate pdfs
        pdf_factor = self._eval_pdfs(x1, x2, pid)

        return pdf_factor * self.partonic_dxs(cos_theta, s_parton, isq)
    
    def single_channel(self, p: tf.Tensor, channel: int):
        # Map input to needed quantities
        x1, x2, cos_theta, _ = tf.unstack(p, axis=-1)
        s_parton = x1 * x2 * self.s_had
        
        # Calculate full weight
        m0u, m1u, m2u, _, _, _ = self.amp2_single(cos_theta, s_parton, 'u')
        m0d, m1d, m2d, _, _, _ = self.amp2_single(cos_theta, s_parton, 'd')
        m = [m0u + m0d, m1u + m1d, m2u + m2d]
        return m[channel]

    def __call__(self, p: tf.Tensor):
        """Calculate the full hadronic event weight including pdfs and
        sum over initial-state flavors.

        Args:
            p (tf.Tensor): Momentum input with shape `(N,4)` and parametrization
                `p = {x1, x2, cos_theta, phi}`

        Returns:
            w (tf.Tensor): Returns weight of the event with shape `(1,)`.
        """
        # Map input to needed quantities
        x1, x2, cos_theta, phi = tf.unstack(p, axis=-1)

        # Calculate full weight
        w = 0
        for isq in self.isq:
            w += self.hadronic_dxs(x1, x2, cos_theta, isq)

        return w

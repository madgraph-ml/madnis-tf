from typing import List
import math as m
import warnings

import tensorflow as tf
from pdfflow import mkPDFs

warnings.filterwarnings("ignore")

# -------------------------------------
# Basic Inputs
# -------------------------------------
MW = 8.041900e01   # W-Boson Mass   8.041900e01 (MG5)
MZ = 9.118800e01   # Z-Boson Mass   9.118800e01 (MG5)
WZ = 2.441404e00   # Z-Boson Width  2.441404e00 (MG5)
GF = 1.166390e-05  # Fermi Constant 1.166390e-05 (MG5)
NC = 3  # Color factor
# -------------------------------------


class DrellYan:
    r"""Amplitudes, weights and differential cross section
    for the Drell-Yan process

        q q~ > Z/gamma > l+ l-

    where q = {u,c,b,d,s}.
    """

    def __init__(
        self,
        isq: List[str],
        e_had: float = 13000.0,
        mw: float = MW,
        mz: float = MZ,
        wz: float = WZ,
        gf: float = GF,
        pdfset: str = "NNPDF40_lo_as_01180",
        input_format: str = "cartesian",
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
            gf (float, optional): Fermi constant. Defaults to GF.
            pdfset (str, optional): PDF-set to use.. Defaults to "NNPDF40_lo_as_01180".
                Requires `lhapdf` and `lhapdf-management`.
            input_format (str,optional): Which parametrization do the momenta have.
                Default is `"cartesian"` which means `p = {px1,py1,pz1,pz2}`. Alternative
                is `"convpolar"` as `p = {x1,x2,costheta,phi}`.
        """
        super().__init__(**kwargs)
        self._dtype = tf.keras.backend.floatx()

        # Input parameters
        self.e_had = e_had
        self.s_had = e_had**2
        self.mw = mw
        self.mz = mz
        self.wz = wz
        self.gf = gf

        # Define input format
        self.input_format = input_format

        # Factorisation scale and pdfset
        self.muf2 = self.mz ** 2
        self.pdf = mkPDFs(pdfset, [0])

        # Basic Definitions
        self.cw2 = self.mw**2 / self.mz**2
        self.sw2 = 1 - self.cw2

        # Definition of electroweak coupling
        #self.alpha = m.sqrt(2) * self.gf * self.mw**2 * self.sw2 # Gf scheme
        self.alpha = 1/1.325070e+02 # MG-Input

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

    def a(self, s: tf.Tensor, mode: str, isq: str):
        if mode == "ZZ":
            m_ZZ = (
                (self.V_q[isq] ** 2 + self.A_q[isq] ** 2)
                * (self.V_l**2 + self.A_l**2)
                / 4
            )
            factor_g2 = 8 / m.sqrt(2) * self.gf * self.mz**2
            return s**2 / 4 * factor_g2**2 * m_ZZ
        elif mode == "yy":
            m_yy = 4 * self.Q_f[isq] ** 2
            factor_e2 = 4 * m.pi * self.alpha
            return s**2 / 4 * factor_e2**2 * m_yy
        else:
            m_yz = (-1) * self.Q_f[isq] * self.V_q[isq] * self.V_l
            factor_e2 = 4 * m.pi * self.alpha
            factor_g2 = 8 / m.sqrt(2) * self.gf * self.mz**2
            return s**2 / 4 * factor_g2 * factor_e2 * m_yz

    def b(self, s: tf.Tensor, mode: str, isq: str):
        if mode == "ZZ":
            m_ZZ = self.A_q[isq] * self.A_l * self.V_q[isq] * self.V_l
            factor_g = 8 / m.sqrt(2) * self.gf * self.mz**2
            return s**2 / 2 * factor_g**2 * m_ZZ
        elif mode == "yy":
            return 0.0
        else:
            m_yz = (-1) * self.Q_f[isq] * self.A_q[isq] * self.A_l
            factor_e = 4 * m.pi * self.alpha
            factor_g = 8 / m.sqrt(2) * self.gf * self.mz**2
            return s**2 / 2 * factor_g * factor_e * m_yz

    def prop_factor(self, s: tf.Tensor, mode: str):
        if mode == "ZZ":
            den = (s - self.mz**2) ** 2 + self.wz**2 * self.mz**2
            return 1 / den
        elif mode == "yy":
            den = s**2
            return 1 / den
        else:
            nom = s - self.mz**2
            den = s * ((s - self.mz**2) ** 2 + self.wz**2 * self.mz**2)
            return nom / den

    def amp2_single(self, cos_theta: tf.Tensor, s: tf.Tensor, mode: str, isq: str):
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
        if mode == "yy":
            return (
                self.prop_factor(s, mode) * self.a(s, mode, isq) * (1 + cos_theta**2)
            )

        m_sym = self.a(s, mode, isq) * (1 + cos_theta**2)
        m_asym = self.b(s, mode, isq) * cos_theta
        n_spins = 2
        return n_spins**2 * self.prop_factor(s, mode) * (m_sym + m_asym)

    def amp2_all(self, cos_theta: tf.Tensor, s: tf.Tensor, isq: str):
        """Full squared matrix element. Note the relative sign
        of the interference contribution.

        Args:
            cos_theta (tf.Tensor): scattering angle in partonic CM frame.
            s (tf.Tensor): partonic CM energy.
            isq (str): initial-state quark.

        Returns:
            m2: Full squared matrix element (|M_Z + M_y|^2)
        """
        m_yy = self.amp2_single(cos_theta, s, "yy", isq)
        m_ZZ = self.amp2_single(cos_theta, s, "ZZ", isq)
        m_int = self.amp2_single(cos_theta, s, "yZ", isq)
        return m_yy + m_ZZ + 2 * m_int #Correct sign?

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
        ps_weight = 1 / (32 * m.pi**2) # TODO: Remove this from amplitude! -> PS-Mapping
        fluxm1 = 1 / (2 * s) # TODO: also remove from amplitude -> Different class CrossSection?
        return fluxm1 * ps_weight * cs_factor * self.amp2_all(cos_theta, s, isq)

    def hadronic_dxs( #TODO: Shift to new class CrossSection?
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
        pid = tf.cast([pid], dtype=tf.int32)

        # Calculate pdfs
        q2 = tf.cast(tf.ones_like(x1) * self.muf2, dtype=self._dtype)
        x1 = tf.cast(x1, dtype=self._dtype)
        x2 = tf.cast(x2, dtype=self._dtype)
        pdf_1 = self.pdf.xfxQ2(pid, x1, q2) / x1
        pdf_2 = self.pdf.xfxQ2(-pid, x2, q2) / x2
        s_parton = x1 * x2 * self.s_had
        return pdf_1 * pdf_2 * self.partonic_dxs(cos_theta, s_parton, isq)

    def _cartesian_det(self, r3, x1, x2):
        s = self.s_had * x1 * x2
        r2 = tf.math.log(x1) / tf.math.log(s / self.s_had)
        det1 = 4 * m.pi * tf.math.log(self.s_had / s) / self.s_had
        det2 = (
            m.pi
            * (s / self.s_had) ** (-2 * r2)
            * (-r3 * s + (-1 + r3) * (s / self.s_had) ** (2 * r2) * self.s_had)
            * (s - r3 * s + r3 * (s / self.s_had) ** (2 * r2) * self.s_had)
            * tf.math.log(s / self.s_had)
            / (4 * self.s_had)
        )
        det = det1 / det2
        return det

    def __call__(self, p: tf.Tensor):
        """Calculate the full hadronic event weight including pdfs and
        sum over initial-state flavors.

        Args:
            p (tf.Tensor): Momentum input with shape `(N,4)` and parametrization
                `p = {px1, py1, pz1, pz2}`

        Returns:
            w (tf.Tensor): Returns weight of the event with shape `(1,)`.
        """
        if self.input_format == "cartesian":
            # Map input to needed quantities
            px1, py1, pz1, pz2 = tf.unstack(p, axis=-1)
            e1 = tf.math.sqrt(px1**2 + py1**2 + pz1**2)
            e2 = tf.math.sqrt(px1**2 + py1**2 + pz2**2)
            pz_tot = pz1 + pz2
            e_tot = e1 + e2
            x1 = (e_tot + pz_tot) / (self.e_had)
            x2 = (e_tot - pz_tot) / (self.e_had)
            r3 = (2 * pz1 / self.e_had + x2) / (x1 + x2)
            cos_theta = 2 * r3 - 1

            # Trafo determinant
            det = self._cartesian_det(r3, x1, x2)

        elif self.input_format == "convpolar":
            # Map input to needed quantities
            x1, x2, cos_theta, phi = tf.unstack(p, axis=-1)

            # Trafo determinant
            det = 1.0
        else:
            raise ValueError('Input format must be either "cartesian" or "convpolar"')

        # Calculat full weight
        w = 0
        for isq in self.isq:
            w += self.hadronic_dxs(x1, x2, cos_theta, isq)

        return tf.constant(det, dtype=self._dtype) * w

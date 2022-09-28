import tensorflow as tf
import math as m
from pdfflow import mkPDFs
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------
# Basic Inputs
# -------------------------------------
MW = 8.038500e01  # W-Boson Mass
MZ = 9.110000e01  # Z-Boson Mass
WZ = 2.441404e00  # Z-Boson Width
GF = 1.16637e-05  # Fermi Constant
NC = 3  # Color factor
# -------------------------------------


class DrellYan:
    """Amplitude, weight and differential cross section
    for the Drell-Yan process

        q q~ > Z/gamma > l+ l-

    where q = {u,c,d,s}.
    """

    def __init__(
        self,
        isq: str,
        e_had: float = 13000.0,
        mw: float = MW,
        mz: float = MZ,
        wz: float = WZ,
        gf: float = GF,
        pdfset: str = "NNPDF40_lo_as_01180",
        **kwargs
    ):
        """
        Args:
            isq (str): Whether the initial state quark is up-quark type or down-quark type.
                up-quark type   : `"u"`, `"c"`,
                down-quark type : `"d"`, `"s"`
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

        # Factorisation scale and pdfset
        self.muf = self.mz
        self.pdf = mkPDFs(pdfset, [0])

        # Basic Definitions
        self.cw2 = self.mw**2 / self.mz**2
        self.sw2 = 1 - self.cw2

        # Define alpha in G_mu scheme (alpha_0 would be 1/1.279000e+02)
        self.alpha = m.sqrt(2) * self.gf * self.mw**2 * self.sw

        # Define pid and initial state quark
        self.isq = isq
        pid = {"d": 1, "u": 2, "s": 3, "c": 4}[self.isq]
        self.pid = tf.cast([pid], dtype=tf.int32)

        # Define quark charges
        self.Q_f = {"d": -1 / 3, "s": -1 / 3, "u": 2 / 3, "c": 2 / 3}[self.isq]

        # Define couplings

        self.V_l = -1 / 2 + 2 * self.sw2
        self.A_l = -1 / 2

        self.V_q = {
            "d": -1 / 2 + 2 / 3 * self.sw2,
            "s": -1 / 2 + 2 / 3 * self.sw2,
            "u": +1 / 2 - 4 / 3 * self.sw2,
            "c": +1 / 2 - 4 / 3 * self.sw2,
        }[self.isq]

        self.A_q = {"d": -1 / 2, "s": -1 / 2, "u": 1 / 2, "c": 1 / 2}[self.isq]

    def chi_1(self, s: float):
        factor = m.sqrt(2) * self.gf * self.mz**2 / (16 * m.pi * self.alpha)
        num = s * (s - self.mz**2)
        den = (s - self.mz**2) ** 2 + self.wz**2 * self.mz**2
        return factor * num / den

    def chi_2(self, s: float):
        factor = m.sqrt(2) * self.gf * self.mz**2 / (16 * m.pi * self.alpha)
        num = s**2
        den = (s - self.mz**2) ** 2 + self.wz**2 * self.mz**2
        return factor**2 * num / den

    def m0(self, s: float):
        m_yy = self.Q_f**2
        m_yZ = -2 * self.Q_f * self.V_l * self.V_q * self.chi_1(s)
        m_ZZ = (
            (self.V_q**2 + self.A_q**2)
            * (self.V_l**2 + self.A_l**2)
            * self.chi_2(s)
        )
        return m_yy + m_yZ + m_ZZ

    def m1(self, s: float):
        m_yZ = -4 * self.chi_1(s) * self.Q_f * self.A_q * self.A_l
        m_ZZ = +8 * self.chi_2(s) * self.A_q * self.A_l * self.V_q * self.V_l
        return m_yZ + m_ZZ

    def amp2(self, cos_theta: float, s: float):
        return (
            16
            * m.pi**2
            * self.alpha**2
            * (self.m0(s) * (1 + cos_theta**2) + self.m1(s) * cos_theta)
        )

    def partonic_weight(self, cos_theta: float, s: float):
        factor = 16 * m.pi**2 * 4 * NC * s
        return self.amp2(cos_theta, s) / factor

    def hadronic_weight(self, x1: tf.Tensor, x2: tf.Tensor, cos_theta: tf.Tensor):
        q2 = tf.ones_like(x1) * self.muf
        pdf_1 = self.pdf.xfxQ2(self.pid, x1, q2) / x1
        pdf_2 = self.pdf.xfxQ2(-self.pid, x2, q2) / x2
        return pdf_1 * pdf_2 * self.partonic_weight(cos_theta, x1 * x2 * self.s_had)

    def call(self, p):
        # Current
        px1, py1, pz1, pz2 = tf.unstack(p, axis=-1)
        e1 = tf.math.sqrt(px1**2 + py1**2 + pz1**2)
        e2 = tf.math.sqrt(px1**2 + py1**2 + pz2**2)
        pz_tot = pz1 + pz2
        e_tot = e1 + e2

        cos_theta = pz1 / e1
        x1 = (e_tot + pz_tot) / (self.e_had)
        x2 = (e_tot - pz_tot) / (self.e_had)

        # alternative?
        # x1, x2, cos_theta, phi = tf.unstack(p, axis=-1)

        w = self.hadronic_weight(x1, x2, cos_theta)
        raise w

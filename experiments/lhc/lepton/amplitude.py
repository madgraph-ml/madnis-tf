from zlib import MAX_WBITS
import tensorflow as tf
import math as m

# Basic Inputs
MW = 8.038500e+01
MZ = 9.110000e+01
WW = 2.047600e+00
WZ = 2.441404e+00
GF = 1.16637e-05
NC = 3

class SquaredAmplitude():
    """Squared amplitude for
    
        e+ e- > Z/gamma > q q~
        
    where q = {u,c,d,s}
    """
    def __init__(
        self,
        fsq: str,
        mw: float = MW,
        mz: float = MZ,
        ww: float = WW,
        wz: float = WZ,
        gf: float = GF,
        **kwargs
    ):
        """
        Args:
            fsq (str): Whether the final state quark is up-quark type or down-quark type.
                up-quark type   : `"u"`, `"c"`,
                down-quark type : `"d"`, `"s"`
        """
        super().__init__(**kwargs)
        self.fsq = fsq
        self._dtype = tf.keras.backend.floatx()
        
        # Input parameters
        self.mw = mw
        self.mz = mz
        self.ww = ww
        self.wz = wz
        self.gf = gf
        
        # Basic Definitions
        self.cw2 = self.mw**2/self.mz**2
        self.sw2 = 1 - self.cw2
        self.alpha = 1/1.279000e+02

        self.V_l = -1/2 + 2 * self.sw2
        self.A_l = -1/2
        
    def Q_f(self):
        if self.fsq == 'u' or self.fsq == 'c':
            return 2/3
        else:
            return -1/3
        
    def V_q(self):
        if self.fsq == 'u' or self.fsq == 'c':
            return +1/2 - 4/3 * self.sw2
        else:
            return -1/2 + 2/3 * self.sw2

    def A_q(self):
        if self.fsq == 'u' or self.fsq == 'c':
            return +1/2
        else:
            return -1/2
        
    def chi_1(self, s: float):
        factor = tf.math.sqrt(2) * self.gf * self.mz**2/(16 * m.pi * self.alpha)
        num = s * (s - self.mz**2)
        den = (s - self.mz**2)**2 + self.wz**2 * self.mz**2
        return factor * num/den

    def chi_2(self, s: float):
        factor = tf.math.sqrt(2) * self.gf * self.mz**2/(16 * m.pi * self.alpha)
        num = s**2
        den = (s - self.mz**2)**2 + self.wz**2 * self.mz**2
        return factor**2 * num/den
    
    def m0(self, s: float):
        m_yy = self.Q_f**2
        m_yZ = - 2 * self.Q_f * self.V_l * self.V_q * self.chi_1(s)
        m_ZZ = (self.V_q**2 + self.A_q**2) * (self.V_l**2 + self.A_l**2) * self.chi_2(s)
        return  m_yy + m_yZ + m_ZZ

    def m1(self, s: float):
        m_yZ = - 4 * self.chi_1(s) * self.Q_f * self.A_q * self.A_l
        m_ZZ = + 8 * self.chi_2(s) * self.A_q * self.A_l * self.V_q * self.V_l
        return  m_yZ + m_ZZ
        
    def amp2(self, cos_theta: float, s: float):
        return 16 * m.pi**2 * self.alpha**2 * (self.m0(s) * (1 + cos_theta**2) + self.m1(s) * cos_theta)
    
    def d_sigma(self, cos_theta: float, s: float):
        factor = 16 * m.pi**2 * 4 * NC * s
        return self.amp2(cos_theta,s)/factor

    def call(self, x, s):
        phi = 2 * m.pi * x[:,0:1]
        cos_theta = 2 * x[:,1:2] - 1
        ps_weight = 4 * m.pi
        w = ps_weight * self.d_sigma(cos_theta, s)
        raise w



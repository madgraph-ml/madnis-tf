import numpy as np
import lhapdf


# Inputs
MW = 8.038500e+01
MZ = 9.110000e+01
WW = 2.047600e+00
WZ = 2.441404e+00
GF = 1.16637e-05


# Basic Definitions
cw2 = MW**2/MZ**2
sw2 = 1 - cw2
alpha = 1/1.279000e+02

V_l = -1/2 + 2 * sw2
A_l = -1/2

# make pdf
pdf = lhapdf.mkPDF("NNPDF30_lo_as_0118_nf_4", 0)

def Q_f(q: str):
    if q == 'u' or q == 'c':
        return 2/3
    else:
        return -1/3

def V_q(q: str):
    if q == 'u' or q == 'c':
        return +1/2 - 4/3 * sw2
    else:
        return -1/2 + 2/3 * sw2

def A_q(q: str):
    if q == 'u' or q == 'c':
        return +1/2
    else:
        return -1/2

def prop_1(s: float):
    factor = np.sqrt(2) * GF * MZ**2/(16 * np.pi * alpha)
    num = s * (s - MZ**2)
    den = (s - MZ**2)**2 + WZ**2 * MZ**2
    return factor * num/den

def prop_2(s: float):
    factor = np.sqrt(2) * GF * MZ**2/(16 * np.pi * alpha)
    num = s**2
    den = (s - MZ**2)**2 + WZ**2 * MZ**2
    return factor**2 * num/den

def a0(s: float, q: str):
    m_yy = Q_f(q)**2
    m_yZ = - 2* Q_f(q) * V_l* V_q(q) * prop_1(s)
    m_ZZ = (V_q(q)**2 + A_q(q)**2) * (V_l**2 + A_l**2) * prop_2(s)
    return  m_yy + m_yZ + m_ZZ

def a1(s: float, q: str):
    m_yZ = - 4 * prop_1(s) * Q_f(q) * A_q(q) * A_l
    m_ZZ = + 8 * prop_2(s) * A_q(q) * A_l * V_q(q) * V_l
    return  m_yZ + m_ZZ

def m2(cos_theta: float, s: float, q: str):
    return 16 * np.pi * alpha**2 * (a0(s,q) * (1 + cos_theta**2) + a1(s,q) * cos_theta)


def d_sigma(cos_theta: float, s: float, q: str):
    return alpha**2/(12 * s) * (a0(s,q) * (1 + cos_theta**2) + a1(s,q) * cos_theta)


def full_weight(x1: float, x2: float, cos_theta: float, s: float, q: str):
    ps_weight = 4 * np.pi
    pdf_1  = pdf.xfxQ(2,  x1, MZ**2)
    pdf_2  = pdf.xfxQ(-2, x2, MZ**2)
    return ps_weight * d_sigma(cos_theta, s, q)


def weighted_events(x, s):
    phi = 2 * np.pi* x[:,0:1]
    cos_theta = 2 * x[:,1:2] - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    ps_weight = 4 * np.pi

    pE1 = np.sqrt(s)/2 * np.ones_like(phi)
    px1 = np.sqrt(s)/2 * sin_theta * np.cos(phi)
    py1 = np.sqrt(s)/2 * sin_theta * np.sin(phi)
    pz1 = np.sqrt(s)/2 * cos_theta

    pE2 = np.sqrt(s)/2 * np.ones_like(phi)
    px2 = - np.sqrt(s)/2 * sin_theta * np.cos(phi)
    py2 = - np.sqrt(s)/2 * sin_theta * np.sin(phi)
    pz2 = - np.sqrt(s)/2 * cos_theta

    # add lhapdfs
    w = ps_weight * d_sigma(cos_theta, s, 'u')

    event = []
    event.append(pE1)
    event.append(px1)
    event.append(py1)
    event.append(pz1)
    event.append(pE2)
    event.append(px2)
    event.append(py2)
    event.append(pz2)
    event.append(w)

    event = np.hstack(event)

    print(event.shape)

    return event



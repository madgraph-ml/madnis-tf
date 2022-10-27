import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

# Use double precision
tf.keras.backend.set_floatx("float64")
CUT = 10
# setting path
dydir = os.path.abspath("../experiments/lhc/drell-yan/")
sys.path.append(dydir)
from fudge_integrand import FudgeDrellYan, MZ, WZ, MZP, WZP
from madnis.mappings.phasespace_2p import TwoParticlePhasespaceB

integrand = FudgeDrellYan(["u", "d"], input_format="convpolar")

mapB0 = TwoParticlePhasespaceB(s_mass=MZ, s_gamma=WZ, sqrt_s_min=CUT)
mapB1 = TwoParticlePhasespaceB(sqrt_s_min=CUT)
mapB2 = TwoParticlePhasespaceB(s_mass=MZP, s_gamma=WZP, sqrt_s_min=CUT)

E_BEAM = 6500

def invariant_mass(p, format="cartesian"):
    if format == "cartesian":
        px1, py1, pz1, pz2 = tf.unstack(p, axis=-1)
        e1 = tf.math.sqrt(px1**2 + py1**2 + pz1**2)
        e2 = tf.math.sqrt(px1**2 + py1**2 + pz2**2)
        e_tot = e1 + e2
        pz_tot = pz1 + pz2
        m = tf.math.sqrt(e_tot**2 - pz_tot**2)
        return m
    elif format == "convpolar":
        x1, x2, costheta, phi = tf.unstack(p, axis=-1)
        m = 2 * E_BEAM * tf.math.sqrt(x1 * x2)
        return m
    else:
        raise ValueError('Input format must be either "cartesian" or "convpolar"')
    

# Sampling 
n_events = int(1e6)

# Events in shape (px1, py1, pz1, pz2)
eventsB0 = mapB0.sample(n_events)
weightB0 =  integrand(eventsB0) / mapB0.prob(eventsB0)
mB0 = invariant_mass(eventsB0,"convpolar")

eventsB1 = mapB1.sample(n_events)
weightB1 = integrand(eventsB1) / mapB1.prob(eventsB1)
mB1 = invariant_mass(eventsB1,"convpolar")

eventsB2 = mapB2.sample(n_events)
weightB2 =  integrand(eventsB2) / mapB2.prob(eventsB2)
mB2 = invariant_mass(eventsB2,"convpolar")

# Define histos
rmax = 650
rmin = CUT
bins = 40

m_b0, x_bins = np.histogram(mB0, bins=np.linspace(rmin,rmax, bins+1), density=True, weights=weightB0.numpy())
m_b1, _ = np.histogram(mB1, bins=np.linspace(rmin,rmax, bins+1), density=True, weights=weightB1.numpy())
m_b2, _ = np.histogram(mB2, bins=np.linspace(rmin,rmax, bins+1), density=True, weights=weightB2.numpy())

# make plot
plt.rc("text", usetex=True)
plt.rc('font', family='serif', size=12.0)
plt.rc('axes', labelsize='large')   
fig, ax1 = plt.subplots()
#ax1.step(x_bins[:bins], m_b0, label=r"Map - Z", linewidth=1.0, where='mid')
#ax1.step(x_bins[:bins], m_b2, label=r"Map - Z'", linewidth=1.0, where='mid')
ax1.step(x_bins[:bins], m_b1, label=r"Map - $\gamma$", linewidth=1.0, where='mid')
ax1.legend(frameon=False)
ax1.set_yscale("log")
ax1.set_xlabel(r"$m_{\mathrm{e}^+\mathrm{e}^-}$ [GeV]")
ax1.set_ylabel(r"$\frac{\mathrm{d}\sigma}{\mathrm{d}m_{\mathrm{e}^+\mathrm{e}^-}}$")
fig.savefig("test_fudge_amplitude.pdf", bbox_inches="tight")
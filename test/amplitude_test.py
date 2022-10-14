import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

# Use double precision
tf.keras.backend.set_floatx("float64")
 
# setting path
dydir = os.path.abspath("../experiments/lhc/drell-yan/")
sys.path.append(dydir)
from dy_integrand import DrellYan, MZ, WZ
from madnis.mappings.phasespace_2p import TwoParticlePhasespaceA, TwoParticlePhasespaceB

integrandA = DrellYan(["u", "d"], input_format="cartesian")
integrandB = DrellYan(["u", "d"], input_format="convpolar")

mapA0 = TwoParticlePhasespaceA(s_mass=MZ, s_gamma=WZ)
mapA1 = TwoParticlePhasespaceA()

mapB0 = TwoParticlePhasespaceB(s_mass=MZ, s_gamma=WZ)
mapB1 = TwoParticlePhasespaceB()

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
eventsA0 = mapA0.sample(n_events)
weightA0 =  integrandA(eventsA0) / mapA0.prob(eventsA0)
mA0 = invariant_mass(eventsA0,"cartesian")

eventsA1 = mapA1.sample(n_events)
weightA1 = integrandA(eventsA1) / mapA1.prob(eventsA1)
mA1 = invariant_mass(eventsA1,"cartesian")

eventsB0 = mapB0.sample(n_events)
weightB0 =  integrandB(eventsB0) / mapB0.prob(eventsB0)
mB0 = invariant_mass(eventsB0,"convpolar")

eventsB1 = mapB1.sample(n_events)
weightB1 = integrandB(eventsB1) / mapB1.prob(eventsB1)
mB1 = invariant_mass(eventsB1,"convpolar")

# Define histos
rmax = 250
rmin = 50
bins = 50

m_a0, x_bins = np.histogram(mA0, bins=np.linspace(rmin,rmax, bins+1), density=True, weights=weightA0.numpy())
m_a1, _ = np.histogram(mA1, bins=np.linspace(rmin,rmax, bins+1), density=True, weights=weightA1.numpy())
m_b0, _ = np.histogram(mB0, bins=np.linspace(rmin,rmax, bins+1), density=True, weights=weightB0.numpy())
m_b1, _ = np.histogram(mB1, bins=np.linspace(rmin,rmax, bins+1), density=True, weights=weightB1.numpy())

# make plot
plt.rc("text", usetex=True)
plt.rc('font', family='serif', size=12.0)
plt.rc('axes', labelsize='large')   
fig, ax1 = plt.subplots()
ax1.step(x_bins[:bins], m_a0, label=r"MapA - Z", linewidth=1.0, where='mid')
ax1.step(x_bins[:bins], m_a1, label=r"MapA - $\gamma$", linewidth=1.0, where='mid')
ax1.step(x_bins[:bins], m_b0, label=r"MapB - Z", linewidth=1.0, where='mid')
ax1.step(x_bins[:bins], m_b1, label=r"MapB - $\gamma$", linewidth=1.0, where='mid')
ax1.legend(frameon=False)
ax1.set_yscale("log")
ax1.set_xlabel(r"$m_{\mathrm{e}^+\mathrm{e}^-}$")
ax1.set_ylabel(r"$\frac{\mathrm{d}\sigma}{\mathrm{d}m_{\mathrm{e}^+\mathrm{e}^-}}$")
fig.savefig("test_amplitude.pdf", bbox_inches="tight")
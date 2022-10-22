import tensorflow as tf

def to_four_mom(x):
    e_beam = 6500
    x1, x2, costheta, phi = tf.unstack(x, axis=1)
    s = 4 * e_beam**2 * x1 * x2
    r3 = (costheta + 1) / 2
    pz1 = e_beam * (x1*r3 + x2*(r3-1))
    pz2 = e_beam * (x1*(1-r3) - x2*r3)
    pt = tf.math.sqrt(s*r3*(1-r3))
    px1 = pt * tf.math.cos(phi)
    py1 = pt * tf.math.sin(phi)
    e1 = tf.math.sqrt(px1**2 + py1**2 + pz1**2)
    e2 = tf.math.sqrt(px1**2 + py1**2 + pz2**2)
    return tf.stack((e1, px1, py1, pz1, e2, -px1, -py1, pz2), axis=-1)
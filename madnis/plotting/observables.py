"""Observables for plotting"""

import numpy as np
import tensorflow as tf
from typing import Union
import math as m


class Observable:
    """
    Contains different functions to calculate 1-dim observables.
    """

    def __init__(self):
        self.epsilon = 1e-20

    #########################
    # Coordinate observables
    #########################

    @staticmethod
    def _coordinate(
        x: Union[np.ndarray, tf.Tensor], entry: int, particle_id: list = []
    ):
        """Parent function giving the ith coordinate of
        the input x

        # Arguments
                input: Array with input data, tensor or numpy array
                entry: the array entry which should be returned
                particle_id: Not needed
        """
        del particle_id
        return x[:, entry]

    @staticmethod
    def identity(x: Union[np.ndarray, tf.Tensor], particle_id: list = []):
        """Simply gives the input back"""
        del particle_id  # not needed
        return x

    @staticmethod
    def ratio(x: Union[np.ndarray, tf.Tensor], entry_id: list = [0, 1]):
        assert len(entry_id) == 2
        return x[:, entry_id[0]] / x[:, entry_id[1]]

    def coord_0(self, x: Union[np.ndarray, tf.Tensor], particle_id: list = []):
        return self._coordinate(x, 0, particle_id=particle_id)

    def coord_1(self, x: Union[np.ndarray, tf.Tensor], particle_id: list = []):
        return self._coordinate(x, 1, particle_id=particle_id)

    def coord_2(self, x: Union[np.ndarray, tf.Tensor], particle_id: list = []):
        return self._coordinate(x, 2, particle_id=particle_id)

    def coord_i(self, x: Union[np.ndarray, tf.Tensor], particle_id: list = []):
        entry = particle_id[0]
        return self._coordinate(x, entry, particle_id=particle_id)

    #########################
    # Kinematic observables
    #########################

    @staticmethod
    def _momentum(x: Union[np.ndarray, tf.Tensor], entry: int, particle_id: list):
        """Parent function giving the ith
        momentum entry of n particles.

        # Arguments
                input: Array with input data, tensor or numpy array
                entry: the momentum entry which should be returned
                particle_id: Integers, particle IDs of n particles
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.

                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        Ps = 0
        for particle in particle_id:
            Ps += x[:, entry + particle * 4]

        return Ps

    def energy(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        return self._momentum(x, 0, particle_id)

    def x_momentum(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        return self._momentum(x, 1, particle_id)

    def x_momentum_over_abs(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        momentum = self.x_momentum(x, particle_id)
        energy = (
            np.abs(self.x_momentum(x, [0]))
            + np.abs(self.x_momentum(x, [1]))
            + np.abs(self.x_momentum(x, [2]))
            + np.abs(self.x_momentum(x, [3]))
        )
        return momentum / energy

    def y_momentum(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        return self._momentum(x, 2, particle_id)

    def y_momentum_over_abs(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        momentum = self.y_momentum(x, particle_id)
        energy = (
            np.abs(self.y_momentum(x, [0]))
            + np.abs(self.y_momentum(x, [1]))
            + np.abs(self.y_momentum(x, [2]))
            + np.abs(self.y_momentum(x, [3]))
        )
        return momentum / energy

    def z_momentum(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        return self._momentum(x, 3, particle_id)

    @staticmethod
    def momentum_product(x: Union[np.ndarray, tf.Tensor], particle_id: list = [0, 1]):

        # Momentum product only defined for 2 momenta!
        assert len(particle_id) == 2

        EE = 1.0
        PPX = 1.0
        PPY = 1.0
        PPZ = 1.0
        for particle in particle_id:
            EE *= x[:, 0 + particle * 4]
            PPX *= x[:, 1 + particle * 4]
            PPY *= x[:, 2 + particle * 4]
            PPZ *= x[:, 3 + particle * 4]

        return EE - PPX - PPY - PPZ

    @staticmethod
    def invariant_mass_square(x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """Squared Invariant Mass.
        This function gives the squared invariant mass of n particles.
        # Arguments
                input: Array with input data
                        that will be used to calculate the invariant mass from.
                particle_id: Integers, particle IDs of n particles.
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.
                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        Es = 0
        PXs = 0
        PYs = 0
        PZs = 0
        for particle in particle_id:
            Es += x[:, 0 + particle * 4]
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        m2 = Es ** 2 - PXs ** 2 - PYs ** 2 - PZs ** 2
        return m2

    def invariant_mass(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """Invariant Mass.
        This function gives the invariant mass of n particles.

        # Arguments
                input: Array with input data
                        that will be used to calculate the invariant mass from.
                particle_id: Integers, particle IDs of n particles.
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.

                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        if isinstance(x, np.ndarray):
            m = np.sqrt(np.clip(self.invariant_mass_square(x, particle_id), 0.0, None))
        elif isinstance(x, tf.Tensor):
            m = tf.math.sqrt(
                tf.maximum(self.invariant_mass_square(x, particle_id), self.epsilon)
            )
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return m

    def transverse_mass(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """
        # Arguments
                input: Array with input data
                        that will be used to calculate the invariant mass from.
                particle_id: Integers, particle IDs of n particles.
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.

                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        Es = 0
        PZs = 0
        for particle in particle_id:
            Es += x[:, 0 + particle * 4]
            PZs += x[:, 1 + particle * 4]

        m2 = Es ** 2 - PZs ** 2
        if isinstance(m2, np.ndarray):
            m = np.sqrt(np.clip(m2, self.epsilon, None))
        elif isinstance(m2, tf.Tensor):
            m = tf.math.sqrt(tf.maximum(m2, self.epsilon))
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return m

    def transverse_momentum(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """This function gives the transverse momentum of n particles.

        # Arguments
                particle_id: Integers, particle IDs of n particles
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.

                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        PXs = 0
        PYs = 0
        for particle in particle_id:
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]

        PXs2 = np.square(PXs)
        PYs2 = np.square(PYs)

        pTs = PXs2 + PYs2
        if isinstance(pTs, np.ndarray):
            m = np.sqrt(np.clip(pTs, self.epsilon, None))
        elif isinstance(pTs, tf.Tensor):
            m = tf.math.sqrt(tf.maximum(pTs, self.epsilon))
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")
        return m

    def rapidity(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """
        This function gives the rapidity y of n particles.

        # Arguments
                particle_id: Integers, particle IDs of n particles
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.

                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        Es = 0
        PZs = 0
        for particle in particle_id:
            Es += x[:, 0 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        if isinstance(x, np.ndarray):
            y = 0.5 * (
                np.log(np.clip(np.abs(Es + PZs), self.epsilon, None))
                - np.log(np.clip(np.abs(Es - PZs), self.epsilon, None))
            )
        elif isinstance(x, tf.Tensor):
            y = 0.5 * (
                tf.math.log(tf.maximum(tf.abs(Es + PZs), self.epsilon))
                - tf.math.log(tf.maximum(tf.abs(Es - PZs), self.epsilon))
            )
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return y

    @staticmethod
    def phi(x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """Azimuthal angle phi.
        This function gives the azimuthal angle oftthe particle.

        # Arguments
                particle_id: Integers, particle IDs of two particles given in
                        the shape:
                        `particle_id = [particle_1, particle_2]`.
        """
        PX1s = 0
        PY1s = 0
        for particle in particle_id:
            PX1s += x[:, 1 + particle * 4]
            PY1s += x[:, 2 + particle * 4]

        if isinstance(x, np.ndarray):
            phi = np.arctan2(PY1s, PX1s)
        elif isinstance(x, tf.Tensor):
            phi = tf.math.atan2(PY1s, PX1s)
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return phi

    def pseudo_rapidity(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """Psudo Rapidity.
        This function gives the pseudo rapidity of n particles.

        # Arguments
                particle_id: Integers, particle IDs of n particles
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.

                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        PXs = 0
        PYs = 0
        PZs = 0
        for particle in particle_id:
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        if isinstance(x, np.ndarray):
            Ps = np.sqrt(np.square(PXs) + np.square(PYs) + np.square(PZs))
            eta = 0.5 * (
                np.log(np.clip(np.abs(Ps + PZs), self.epsilon, None))
                - np.log(np.clip(np.abs(Ps - PZs), self.epsilon, None))
            )
        elif isinstance(x, tf.Tensor):
            Ps = tf.math.sqrt(PXs ** 2 + PYs ** 2 + PZs ** 2)
            eta = 0.5 * (
                tf.math.log(tf.maximum(tf.abs(Ps + PZs), self.epsilon))
                - tf.math.log(tf.maximum(tf.abs(Ps - PZs), self.epsilon))
            )
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return eta

    def pseudo_rapidity_cut(
        self, x: Union[np.ndarray, tf.Tensor], particle_id: list, cut: float = 6.0
    ):
        """Psudo Rapidity.
        This function gives the pseudo rapidity of n particles.

        # Arguments
                particle_id: Integers, particle IDs of n particles
                        If there is only one momentum that has to be considered
                        the shape is:
                        `particle_id = [particle_1]`.

                        If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                        `particle_id = [particle_1, particle_2,..]`.
        """
        PXs = 0
        PYs = 0
        PZs = 0
        for particle in particle_id:
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        if isinstance(x, np.ndarray):
            Ps = np.sqrt(np.square(PXs) + np.square(PYs) + np.square(PZs))
            eta = 0.5 * (
                np.log(np.clip(np.abs(Ps + PZs), self.epsilon, None))
                - np.log(np.clip(np.abs(Ps - PZs), self.epsilon, None))
            )
            trans = np.log((cut + eta) / (cut - eta))
        elif isinstance(x, tf.Tensor):
            Ps = tf.math.sqrt(PXs ** 2 + PYs ** 2 + PZs ** 2)
            eta = 0.5 * (
                tf.math.log(tf.maximum(tf.abs(Ps + PZs), self.epsilon))
                - tf.math.log(tf.maximum(tf.abs(Ps - PZs), self.epsilon))
            )
            trans = tf.math.log((cut + eta) / (cut - eta))
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return trans

    def delta_phi(self, x: Union[np.ndarray, tf.Tensor], particle_id: list):
        """Delta Phi.
        This function gives the difference in the azimuthal angle of 2 particles.

        # Arguments
                particle_id: Integers, particle IDs of two particles given in
                        the shape:
                        `particle_id = [particle_1, particle_2]`.
        """
        # Only valid for two momenta
        assert len(particle_id) == 2

        phi1s = self.phi(x, particle_id=[particle_id[0]])
        phi2s = self.phi(x, particle_id=[particle_id[1]])

        if isinstance(x, np.ndarray):
            dphi = np.fabs(phi1s - phi2s)
            dphimin = np.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)
        elif isinstance(x, tf.Tensor):
            dphi = tf.abs(phi1s - phi2s)
            dphimin = tf.where(dphi > m.pi, 2.0 * m.pi - dphi, dphi)
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return dphimin

    def delta_rapidity(
        self, x: Union[np.ndarray, tf.Tensor], particle_id: list = [0, 1]
    ):
        """Delta Rapidity.
        This function gives the rapidity difference of 2 particles.

        # Arguments
                particle_id: Integers, particle IDs of two particles given in
                        the shape:
                        `particle_id = [particle_1, particle_2]`.
        """
        # Only valid for two momenta
        assert len(particle_id) == 2

        y1 = self.pseudo_rapidity(x, particle_id=[particle_id[0]])
        y2 = self.pseudo_rapidity(x, particle_id=[particle_id[1]])

        if isinstance(x, np.ndarray):
            dy = np.abs(y1 - y2)
        elif isinstance(x, tf.Tensor):
            dy = tf.abs(y1 - y2)
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return dy

    def delta_R(self, x: Union[np.ndarray, tf.Tensor], particle_id: list = [0, 1]):
        """Delta R.
        This function gives the Delta R of 2 particles.
        # Arguments
                particle_id: Integers, particle IDs of two particles given in
                        the shape:
                        `particle_id = [particle_1, particle_2]`.
        """
        # Only valid for two momenta
        assert len(particle_id) == 2

        dy = self.delta_rapidity(x, particle_id=particle_id)
        dphi = self.delta_phi(x, particle_id=particle_id)

        if isinstance(x, np.ndarray):
            dR = np.sqrt(dphi ** 2 + dy ** 2)
        elif isinstance(x, tf.Tensor):
            dR = tf.math.sqrt(dphi ** 2 + dy ** 2)
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return dR

    def log_delta_R(
        self,
        x: Union[np.ndarray, tf.Tensor],
        particle_id: list = [0, 1],
        cut: float = 0.4,
    ):
        """
        This function gives the log transformed Delta R of 2 particles
        without the cut.
        # Arguments
                particle_id: Integers, particle IDs of two particles given in
                        the shape:
                        `particle_id = [particle_1, particle_2]`.
        """
        assert len(particle_id) == 2
        dR = self.delta_R(x, particle_id=particle_id)

        if isinstance(dR, np.ndarray):
            transform = np.log(dR - cut)
        elif isinstance(x, tf.Tensor):
            transform = tf.math.log(dR - cut)
        else:
            raise ValueError("Input is not a valid tensor type (torch or numpy)")

        return transform

from typing import Tuple
import tensorflow as tf
from typing import Dict
import numpy as np

# Import Model blocks
from ..distributions.uniform import StandardUniform
from ..mappings.flow import Flow
from ..transforms.coupling.all_in_one_block import AllInOneBlock
from ..transforms.coupling.coupling_splines import RationalQuadraticSplineBlock
from ..transforms import permutation as perm
from ..transforms.nonlinearities import Sigmoid, Logit
from ..transforms.coupling.coupling_linear import AffineCoupling
from ..transforms.actnorm import ActNorm
from .utils import binary_list


class AllInOneVegasFlow(Flow):
    """Defines the vegas flow network
    using the all in one block from FrEIA"""

    def __init__(
        self,
        dims_in: Tuple[int],
        n_blocks: int,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: callable = None,
        hypercube_target: bool = False,
        name="VegasFlow",
        **kwargs,
    ):

        self.dims_in = dims_in
        self.dims_c = dims_c

        # Define base_dist
        base_dist = StandardUniform(dims_in)

        # Define transforms
        transforms = []
        if hypercube_target:
            transforms.append(Logit(dims_in))
        for _ in range(n_blocks):
            transforms.append(
                AllInOneBlock(
                    self.dims_in,
                    dims_c=self.dims_c,
                    clamp=0.5,
                    permute_soft=True,
                    subnet_meta=subnet_meta,
                    subnet_constructor=subnet_constructor,
                )
            )
        transforms.append(Sigmoid(dims_in))

        super().__init__(base_dist, transforms, embedding_net=None, name=name, **kwargs)


class AffineVegasFlow(Flow):
    """Defines the vegas flow network
    with various permuation possibilities"""

    def __init__(
        self,
        dims_in: Tuple[int],
        n_blocks: int,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: callable = None,
        hypercube_target: bool = False,
        permutations: str = "random",
        name="VegasFlow",
        **kwargs,
    ):

        self.dims_in = dims_in
        self.dims_c = dims_c

        # setting up permutations
        if permutations == "exchange":
            perm_list = [
                perm.PermuteExchange(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        elif permutations == "random":
            perm_list = [
                perm.PermuteRandom(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        elif permutations == "log":
            # taken from i-flow: https://gitlab.com/i-flow/i-flow/-/blob/master/iflow_test.py#L365

            n_perms = int(np.ceil(np.log2(self.dims_in)))
            # use at least n_perms blocks
            n_blocks = int(2 * n_perms)
            masks = np.transpose(
                np.array([binary_list(i, n_perms) for i in range(self.dims_in[0])])
            )[::-1]
            # now find perm that moves all '1' of masks to the end
            perm_list = []
            for i, mask in enumerate(masks[::-1]):
                mask = mask.astype(bool)
                if i == 0:
                    # first perm. starts from ordered dimensions
                    permutation = np.concatenate(
                        [
                            np.arange(self.dims_in[0])[mask],
                            np.arange(self.dims_in[0])[~mask],
                        ]
                    )
                else:
                    # subsequent perm. start from previously permuted (and exchanged) dimensions
                    previous_perm_corr = np.arange(self.dims_in[0])[
                        np.argsort(permutation)
                    ]
                    permutation = np.concatenate(
                        [
                            previous_perm_corr[np.arange(self.dims_in[0])][mask],
                            previous_perm_corr[np.arange(self.dims_in[0])][~mask],
                        ]
                    )
                perm_list.append(
                    perm.Permutation(
                        self.dims_in, dims_c=self.dims_c, permutation=permutation
                    )
                )
                perm_list.append(perm.PermuteExchange(self.dims_in, dims_c=self.dims_c))

        elif permutations == "soft":
            perm_list = [
                perm.PermuteSoft(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        elif permutations == "softlearn":
            perm_list = [
                perm.PermuteSoftLearn(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        else:
            raise ValueError("Permutation '{}' not recognized".format(permutations))

        # Define base_dist
        base_dist = StandardUniform(dims_in)

        # Define transforms
        transforms = []
        if hypercube_target:
            transforms.append(Logit(dims_in))
        for i in range(n_blocks):
            transforms.append(
                AffineCoupling(
                    self.dims_in,
                    dims_c=self.dims_c,
                    subnet_meta=subnet_meta,
                    subnet_constructor=subnet_constructor,
                    clamp=0.5,
                )
            )
            transforms.append(ActNorm(self.dims_in, dims_c=self.dims_c))
            transforms.append(perm_list[i])

        transforms.append(Sigmoid(dims_in))

        super().__init__(base_dist, transforms, embedding_net=None, name=name, **kwargs)


class RQSVegasFlow(Flow):
    """Defines the vegas flow network
    with RQ splines like i-flow"""

    def __init__(
        self,
        dims_in: Tuple[int],
        n_blocks: int,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: callable = None,
        hypercube_target: bool = False,
        bins: int = 8,
        permutations: str = "random",
        name="RQSVegasFlow",
        **kwargs,
    ):

        self.dims_in = dims_in
        self.dims_c = dims_c

        # setting up permutations
        if permutations == "exchange":
            perm_list = [
                perm.PermuteExchange(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        elif permutations == "random":
            perm_list = [
                perm.PermuteRandom(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        elif permutations == "log":
            # taken from i-flow: https://gitlab.com/i-flow/i-flow/-/blob/master/iflow_test.py#L365

            n_perms = int(np.ceil(np.log2(self.dims_in)))
            n_blocks = int(2 * n_perms)
            masks = np.transpose(
                np.array([binary_list(i, n_perms) for i in range(self.dims_in[0])])
            )[::-1]
            # now find perm that moves all '1' of masks to the end
            perm_list = []
            for i, mask in enumerate(masks[::-1]):
                mask = mask.astype(bool)
                if i == 0:
                    # first perm. starts from ordered dimensions
                    permutation = np.concatenate(
                        [
                            np.arange(self.dims_in[0])[mask],
                            np.arange(self.dims_in[0])[~mask],
                        ]
                    )
                else:
                    # subsequent perm. start from previously permuted (and exchanged) dimensions
                    previous_perm_corr = np.arange(self.dims_in[0])[
                        np.argsort(permutation)
                    ]
                    permutation = np.concatenate(
                        [
                            previous_perm_corr[np.arange(self.dims_in[0])][mask],
                            previous_perm_corr[np.arange(self.dims_in[0])][~mask],
                        ]
                    )
                perm_list.append(
                    perm.Permutation(
                        self.dims_in, dims_c=self.dims_c, permutation=permutation
                    )
                )
                perm_list.append(perm.PermuteExchange(self.dims_in, dims_c=self.dims_c))

        elif permutations == "soft":
            perm_list = [
                perm.PermuteSoft(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        elif permutations == "softlearn":
            perm_list = [
                perm.PermuteSoftLearn(self.dims_in, dims_c=self.dims_c)
                for _ in range(n_blocks)
            ]
        else:
            raise ValueError("Permutation '{}' not recognized".format(permutations))

        # Define base_dist
        base_dist = StandardUniform(dims_in)

        # Define transforms, starting from target side
        transforms = []
        if not hypercube_target:
            transforms.append(Sigmoid(dims_in))
        for i in range(n_blocks):
            transforms.append(
                RationalQuadraticSplineBlock(
                    self.dims_in,
                    dims_c=self.dims_c,
                    subnet_meta=subnet_meta,
                    subnet_constructor=subnet_constructor,
                    num_bins=bins,
                )
            )
            if "soft" in permutations:
                transforms.append(Logit(dims_in))
                transforms.append(perm_list[i])
                transforms.append(Sigmoid(dims_in))
            else:
                transforms.append(perm_list[i])

        # Remove last shuffle as it is useless (only do when no trainables)
        if "soft" not in permutations:
            transforms.pop()

        super().__init__(base_dist, transforms, embedding_net=None, name=name, **kwargs)

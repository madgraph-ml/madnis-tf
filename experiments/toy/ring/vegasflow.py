from typing import Tuple
import tensorflow as tf
from typing import Dict

# Import Model blocks
from madnis.distributions.base import Distribution
from madnis.distributions.uniform import StandardUniform
from madnis.mappings.flow import Flow
from madnis.transforms.all_in_one_block import AllInOneBlock
from madnis.transforms.nonlinearities import Sigmoid, Logit


class VegasFlow(Flow):
    """Defines the vegas flow network"""

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
                    clamp=0.2,
                    permute_soft=True,
                    subnet_meta=subnet_meta,
                    subnet_constructor=subnet_constructor,
                )
            )
        transforms.append(Sigmoid(dims_in))

        super().__init__(base_dist, transforms, embedding_net=None, name=name, **kwargs)

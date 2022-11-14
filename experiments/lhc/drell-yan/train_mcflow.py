from madnis.mappings.phasespace_2p import TwoParticlePhasespaceB
from madnis.models.mc_prior import WeightPrior

from dy_integrand import DrellYan, MZ
from madnis_training import MadnisTraining

class FudgeDrellYanMadnis(MadnisTraining):
    def define_physics_arguments(self, parser):
        # physics model-parameters
        parser.add_argument("--z_width_scale", type=float, default=1)

        # prior and mapping setting
        parser.add_argument("--cut", type=float, default=15)
        parser.add_argument("--prior", type=str, default="mg5",
                            choices={"mg5", "sherpa", "flat"})
        parser.add_argument("--maps", type=str, default="y",
                            choices={"y", "z", "zy"})

        parser.add_argument("--run_name", type=str)

    def define_integrand(self):
        self.n_channels = len(self.args.maps)
        self.dims_in = 4  # dimensionality of data space

        Z_SCALE = self.args.z_width_scale
        self.WZ = 2.441404e-00 * Z_SCALE

        self.integrand = DrellYan(
            ["u", "d", "c", "s"],
            input_format="convpolar",
            wz=self.WZ,
            z_scale=Z_SCALE
        )

        print(f"\n Integrand specifications:")
        print("-----------------------------------------------------------")
        print(f" Dimensions : {self.dims_in}                              ")
        print(f" Channels   : {self.n_channels}                           ")
        print(f" Z-Width    : {self.WZ} GeV                               ")
        print("-----------------------------------------------------------\n")

    def define_mappings(self):
        self.map_Z = TwoParticlePhasespaceB(s_mass=MZ, s_gamma=self.WZ,
                                            sqrt_s_min=self.args.cut)
        self.map_y = TwoParticlePhasespaceB(sqrt_s_min=self.args.cut, nu=2)

        map_dict = {"y": self.map_y, "z": self.map_Z}
        self.mappings = [map_dict[m] for m in self.args.maps]

    def define_prior(self):
        prior = None
        if self.args.prior == "mg5" and self.args.maps == "zy":
            y_mg_prior = lambda p: self.integrand.single_channel(p, 0)
            z_mg_prior = lambda p: self.integrand.single_channel(p, 1)
            prior = WeightPrior([z_mg_prior, y_mg_prior], self.n_channels)
        elif self.args.prior == "sherpa" and self.args.maps == "zy":
            prior = WeightPrior([map_Z.prob, map_y.prob], self.n_channels)
        self.prior = None if prior is None else prior.get_prior_weights

    def define_output(self):
        mode = "separate" if self.args.separate_flows else "cond"
        alphas = "trained" if self.args.train_mcw else "fixed"
        if self.args.run_name is None:
            self.log_dir = (
                f"./plots/sm/{self.n_channels}channels_{self.args.maps}map_{mode}_" +
                f"{self.args.prior}_{alphas}/"
            )
        else:
            self.log_dir = f"./plots/sm/{self.args.run_name}/"
        self.plot_name = "drell_yan"

FudgeDrellYanMadnis().run()

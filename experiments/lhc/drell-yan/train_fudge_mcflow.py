from madnis.mappings.phasespace_2p import TwoParticlePhasespaceB
from madnis.models.mc_prior import WeightPrior

from fudge_integrand import FudgeDrellYan, MZ, MZP, WZP
from madnis_training import MadnisTraining

class DrellYanMadnis(MadnisTraining):
    def define_physics_arguments(self, parser):
        # physics model-parameters
        parser.add_argument("--z_width_scale", type=float, default=1)
        parser.add_argument("--zp_width_scale", type=float, default=1)

        # prior and mapping setting
        parser.add_argument("--cut", type=float, default=15)
        parser.add_argument("--prior", type=str, default="mg5",
                            choices={"mg5", "sherpa", "flat"})
        parser.add_argument("--maps", type=str, default="y",
                            choices={"y", "z", "p", "zy", "py", "pz", "pzy"})

        parser.add_argument("--run_name", type=str)

    def define_integrand(self):
        self.n_channels = len(self.args.maps)
        self.dims_in = 4  # dimensionality of data space

        Z_SCALE = self.args.z_width_scale
        self.WZ = 2.441404e-00 * Z_SCALE

        ZP_SCALE = self.args.zp_width_scale
        self.WZP = 5.000000e-01 * ZP_SCALE

        self.integrand = FudgeDrellYan(
            ["u", "d", "c", "s"],
            input_format="convpolar",
            wz=self.WZ,
            wzp=self.WZP,
            z_scale=Z_SCALE,
            zp_scale=ZP_SCALE,
        )

        print(f"\n Integrand specifications:")
        print("-----------------------------------------------------------")
        print(f" Dimensions : {self.dims_in}                              ")
        print(f" Channels   : {self.n_channels}                           ")
        print(f" Z-Width    : {self.WZ} GeV                               ")
        print(f" Z'-Width   : {self.WZP} GeV                              ")
        print("-----------------------------------------------------------\n")

    def define_mappings(self):
        self.map_p = TwoParticlePhasespaceB(s_mass=MZP, s_gamma=self.WZP,
                                            sqrt_s_min=self.args.cut)
        self.map_Z = TwoParticlePhasespaceB(s_mass=MZ, s_gamma=self.WZ,
                                            sqrt_s_min=self.args.cut)
        self.map_y = TwoParticlePhasespaceB(sqrt_s_min=self.args.cut, nu=2)

        map_dict = {"y": self.map_y, "z": self.map_Z, "p": self.map_p}
        self.mappings = [map_dict[m] for m in self.args.maps]

    def define_prior(self):
        prior = None
        if self.args.prior == "mg5":
            y_mg_prior = lambda p: self.integrand.single_channel(p, 0)
            z_mg_prior = lambda p: self.integrand.single_channel(p, 1)
            p_mg_prior = lambda p: self.integrand.single_channel(p, 2)
            if self.args.maps == "zy":
                prior = WeightPrior([z_mg_prior, y_mg_prior], self.n_channels)
            elif self.args.maps == "py":
                prior = WeightPrior([p_mg_prior, y_mg_prior], self.n_channels)
            elif self.args.maps == "pz":
                prior = WeightPrior([p_mg_prior, z_mg_prior], self.n_channels)
            elif self.args.maps == "pzy":
                prior = WeightPrior([p_mg_prior, z_mg_prior, y_mg_prior], self.n_channels)
        elif self.args.prior == "sherpa":
            if self.args.maps == "zy":
                prior = WeightPrior([self.map_Z.prob, self.map_y.prob], self.n_channels)
            elif self.args.maps == "py":
                prior = WeightPrior([self.map_p.prob, self.map_y.prob], self.n_channels)
            elif self.args.maps == "pz":
                prior = WeightPrior([self.map_p.prob, self.map_Z.prob], self.n_channels)
            elif self.args.maps == "pzy":
                prior = WeightPrior([self.map_p.prob, self.map_Z.prob, self.map_y.prob],
                                    self.n_channels)

        self.prior = None if prior is None else prior.get_prior_weights

    def define_output(self):
        mode = "separate" if self.args.separate_flows else "cond"
        alphas = "trained" if self.args.train_mcw else "fixed"
        if self.args.run_name is None:
            self.log_dir = (
                f"./plots/zprime/{self.n_channels}channels_{self.args.maps}map_{mode}_" +
                f"{self.args.prior}_{alphas}/"
            )
        else:
            self.log_dir = f"./plots/zprime/{self.args.run_name}/"
        self.plot_name = "fudge_drell_yan"

DrellYanMadnis().run()

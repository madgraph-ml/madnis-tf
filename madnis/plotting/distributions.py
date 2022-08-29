""" Distribution class """

from typing import List
import numpy as np

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .plots import plot_2d_distribution, plot_2d_distribution_single, plot_distribution_ratio, plot_distribution_diff_ratio
from .observables import Observable


class Distribution(Observable):
	"""Custom Distribution class.

	Defines which Observables will be plotted depending on the
	specified dataset.
	"""
	def __init__(
    	self,
		real_data: np.ndarray,
		gen_data: np.ndarray,
		name: str,
		log_dir: str,
		dataset: str,
		latent: bool=False,
		weights: np.ndarray=None,
		which_plots: List[bool]=[True, False, False, True],
    ):
		super(Distribution, self).__init__()
		self.real_data = real_data
		self.gen_data = gen_data
		self.weights = weights
    
		self.name = name
		self.log_dir = log_dir
		self.dataset = dataset
  
		self.latent = latent
		self.which_plots = which_plots

	def plot(self):
		if self.latent == True:
			self.latent_distributions()
		else:
			if self.dataset == 'wp_2j':
				self.w_2jets_distributions()
			elif self.dataset == 'ring':
				self.ring_distributions()
			else:
				self.basic_2d_distributions()

		# pylint: disable=W0702
		try:
			plt.rc("text", usetex=True)
			plt.rc("font", family="serif")
			plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
		except:
			print("No latex installed")

		if self.which_plots[0]:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_ratio.pdf') as pp:
				for observable in self.args.keys():
					fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios' : [4, 1], 'hspace' : 0.00}, figsize=(6.6,6))
					plot_distribution_ratio(axs, self.real_data, self.gen_data, self.weights, self.args[observable])
					fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
					plt.close()
     
		if self.which_plots[1]:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_diff_ratio.pdf') as pp:
				for observable in self.args.keys():
					fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios' : [2, 1, 1], 'hspace' : 0.00}, figsize=(6.6,6))
					plot_distribution_diff_ratio(fig, axs, self.real_data, self.gen_data, self.weights, self.args[observable])
					fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
					plt.close()

		if self.which_plots[2]:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_2d.pdf') as pp:
				for i, observable in enumerate(list(self.args2.keys())):
					for observable2 in list(self.args2.keys())[i+1:]:
						fig, axs = plt.subplots(1,3, figsize=(20,6))
						plot_2d_distribution(fig, axs, self.real_data, self.gen_data, self.weights, self.args[observable], self.args2[observable2])
						plt.subplots_adjust(wspace=0.45, hspace=0.25)
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()
      
		if self.which_plots[3]:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_2d_single.pdf') as pp:
				for i, observable in enumerate(list(self.args2.keys())):
					for observable2 in list(self.args2.keys())[i+1:]:
						fig, axs = plt.subplots(1, figsize=(6.6,6))
						plot_2d_distribution_single(fig, axs, self.real_data, self.gen_data, self.weights, self.args[observable], self.args2[observable2])
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()
      
	def basic_2d_distributions(self):
		# Particle_id (always 0 in this case), observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coord_0, 100, (-1.2,1.2) ,r'$x$', r'$x$',False),
			'y' : ([0], self.coord_1, 100, (-1.2,1.2) ,r'$y$', r'$y$',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coord_0, 100, (-0.6,0.6) ,r'$x$', r'$x$',False),
			'y' : ([0], self.coord_1, 100, (-0.6,0.6) ,r'$y$', r'$y$',False),
		}
	 
		self.args = args
		self.args2 = args2

	def ring_distributions(self):
		# Particle_id (always 0 in this case), observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coord_0, 100, (-2,2) ,r'$x$', r'$x$',False),
			'y' : ([0], self.coord_1, 100, (-2,2) ,r'$y$', r'$y$',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coord_0, 100, (-2,2) ,r'$x$', r'$x$',False),
			'y' : ([0], self.coord_1, 100, (-2,2) ,r'$y$', r'$y$',False),
		}
	 
		self.args = args
		self.args2 = args2

	def w_2jets_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{\mathrm{T}, \mathrm{W}}$ [GeV]', r'p_{\mathrm{T}, \mathrm{W}}',False),
			'pxW' : ([0], self.x_momentum, 50, (-160,160), r'$p_{x, \mathrm{W}}$ [GeV]', r'p_{x, \mathrm{W}}',False),
			'pyW' : ([0], self.y_momentum, 50, (-160,160), r'$p_{y, \mathrm{W}}$ [GeV]', r'p_{y, \mathrm{W}}',False),
			'pzW' : ([0], self.z_momentum, 50, (-600,600), r'$p_{z, \mathrm{W}}$ [GeV]', r'p_{z, \mathrm{W}}',False),
			'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{\mathrm{W}}$ [GeV]', r'E_{\mathrm{W}}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_1}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_1}',False),
			'pxj1' : ([1], self.x_momentum, 40, (-120,120), r'$p_{x, \mathrm{j}_1}$ [GeV]', r'p_{x, \mathrm{j}_1}',False),
			'pyj1' : ([1], self.y_momentum, 40, (-120,120), r'$p_{y, \mathrm{j}_1}$ [GeV]', r'p_{y, \mathrm{j}_1}',False),
			'pzj1' : ([1], self.z_momentum, 50, (-400,400), r'$p_{z, \mathrm{j}_1}$ [GeV]', r'p_{z, \mathrm{j}_1}',False),
			'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{\mathrm{j}_1}$ [GeV]', r'E_{\mathrm{j}_1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_2}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_2}',False),
			'pxj2' : ([2], self.x_momentum, 40, (-120,120), r'$p_{x, \mathrm{j}_2}$ [GeV]', r'p_{x, \mathrm{j}_2}',False),
			'pyj2' : ([2], self.y_momentum, 40, (-120,120), r'$p_{y, \mathrm{j}_2}$ [GeV]', r'p_{y, \mathrm{j}_2}',False),
			'pzj2' : ([2], self.z_momentum, 50, (-400,400), r'$p_{z, \mathrm{j}_2}$ [GeV]', r'p_{z, \mathrm{j}_2}',False),
			'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{\mathrm{j}_2}$ [GeV]', r'E_{\mathrm{j}_2}',False),
			#---------------------#
   			'dPhijj' : ([1,2], self.delta_phi, 40, (0,3.14) ,r'$\Delta\phi_{\mathrm{j}\mathrm{j}}$', r'\Delta\phi_{\mathrm{j}\mathrm{j}}',False),
			'dEtajj' : ([1,2], self.delta_rapidity, 40, (0,5) ,r'$\Delta\eta_{\mathrm{j}\mathrm{j}}$', r'\Delta\eta_{\mathrm{j}\mathrm{j}}',False),
			'dRjj' : ([1,2], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}\mathrm{j}}$', r'\Delta R_{\mathrm{j}\mathrm{j}}',False),
		 	'mwjj' : ([0,1,2], self.invariant_mass, 50, (0,2000), r'$M_{\mathrm{W}\mathrm{j}\mathrm{j}}$ [GeV]', r'p_{x, j2}',False),
		}	 

		args2 = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{\mathrm{T}, \mathrm{W}}$ [GeV]', r'p_{\mathrm{T}, \mathrm{W}}',False),
			'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{\mathrm{W}}$ [GeV]', r'E_{\mathrm{W}}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_1}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_1}',False),
			'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{\mathrm{j}_1}$ [GeV]', r'E_{\mathrm{j}_1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{\mathrm{T}, \mathrm{j}_2}$ [GeV]', r'p_{\mathrm{T}, \mathrm{j}_2}',False),
			'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{\mathrm{j}_2}$ [GeV]', r'E_{\mathrm{j}_2}',False),
			#---------------------#
   			'dPhijj' : ([1,2], self.delta_phi, 40, (0,3.14) ,r'$\Delta\phi_{\mathrm{j}\mathrm{j}}$', r'\Delta\phi_{\mathrm{j}\mathrm{j}}',False),
			'dEtajj' : ([1,2], self.delta_rapidity, 40, (0,5) ,r'$\Delta\eta_{\mathrm{j}\mathrm{j}}$', r'\Delta\eta_{\mathrm{j}\mathrm{j}}',False),
			'dRjj' : ([1,2], self.delta_R, 40, (0,8) ,r'$\Delta R_{\mathrm{j}\mathrm{j}}$', r'\Delta R_{\mathrm{j}\mathrm{j}}',True),
		 	'mjj' : ([1,2], self.invariant_mass, 40, (0,1000), r'$M_{\mathrm{j}\mathrm{j}}$ [GeV]', r'p_{x, j2}',True),
			#---------------------#			
		}	 
	 
		self.args = args
		self.args2 = args2

	def latent_distributions(self):
		 # Particle_id (always 0 in this case), observable, bins, range, x_label, log_scale

		args = {			 
			'z0' : ([0], self.coord_0, 60, (-4,4) ,r'$z_0$', r'$z_0$',False),
			'z1' : ([0], self.coord_1, 60, (-4,4) ,r'$z_1$', r'$z_1$',False),
			#'z2' : ([0], self.coord_2, 60, (-4,4) ,r'$z_2$', r'z_2',False),
			#'z3' : ([0], self.coord_3, 60, (-4,4) ,r'$z_3$', r'z_3',False),
			#'z4' : ([0], self.coord_4, 60, (-4,4) ,r'$z_4$', r'z_4',False),
			#'z5' : ([0], self.coord_5, 60, (-4,4) ,r'$z_5$', r'z_5',False),
			#'z6' : ([0], self.coord_6, 60, (-4,4) ,r'$z_6$', r'z_6',False),
			#'z7' : ([0], self.coord_7, 60, (-4,4) ,r'$z_7$', r'z_7',False),
			#'z8' : ([0], self.coord_8, 60, (-4,4) ,r'$z_8$', r'z_8',False),
			#'z9' : ([0], self.coord_9, 60, (-4,4) ,r'$z_9$', r'z_9',False),
			#'z10' : ([0], self.coord_10, 60, (-4,4) ,r'$z_{10}$', r'z_{10}',False),
			#'z11' : ([0], self.coord_11, 60, (-4,4) ,r'$z_{11}$', r'z_{11}',False),
		}
	 
		self.args = args
		self.args2 = args

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.use('Agg')
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

psi = r'$\Psi = \displaystyle\frac{8\pi e}{3} \left(\frac{2 m_e k_B}{(2\pi\hbar)^2}\right)^{3/2}(r+3/2)F_{(r+1/2)}$'
seebeck = r'$\alpha = \displaystyle\frac{k_B}{e}\left[\frac{(r + 5/2) F_{(r+3/2)}(\eta)}{(r+3/2)F_{(r+1/2)}(\eta)} - \eta \right]$'
seebeck_fermi = r'$\alpha - \displaystyle\frac{k_B}{e}\left[\frac{(r + 5/2) F_{(r+3/2)}(\eta)}{(r+3/2)F_{(r+1/2)}(\eta)} - \eta \right] = 0$'
carrier = r'$\displaystyle n_H = \frac{1}{eR_H} = \frac{(2m^*k_BT)^{3/2}}{3\pi^2\hbar^3} \frac{(r+3/2)^2F_{(r+1/2)}^2(\eta)}{(2r+3/2)F_{(2r+1/2)}(\eta)}$'
lorenz = r'$\displaystyle L = \left(\frac{k_B}{e}\right)^2 \left\{\frac{(r+7/2)F_{(r+5/2)}(\eta)}{(r+3/2)F_{(r+3/2)}(\eta)} - \left[\frac{(r+5/2)F_{(r+3/2)}(\eta)}{(r+3/2)F_{(r+1/2)}(\eta)} \right]\right\}$'
efm = r'$m^* = \displaystyle\frac{1}{2k_B T} \cdot \left(3\pi^2\hbar^3n_H \cdot \frac{(2r+3/2)F_{(2r+1/2)}(\eta)}{(r+3/2)^2 F_{r+1/2}^2 (\eta)}\right)^{2/3}$'
zt = r'$zT = \displaystyle\frac{\alpha^2 T}{\rho \kappa} = \frac{S^2 T}{\rho\kappa}$'
beta = r'$\beta = \displaystyle\frac{\mu_0 (m^*/m_e)^{3/2}T^{5/2}}{\kappa_L}$'
tzt = r'$zT = \displaystyle\frac{\alpha^2}{L + (\Psi\beta)^{-1}}$'
fermi = r'$F_j(\eta) = \displaystyle\int\displaylimits_0^\infty \frac{\epsilon^j d\epsilon}{1+\exp(\epsilon - \eta)}$'
fermi_example = r'$F_{(2r+1)}(\eta) = \displaystyle\int\displaylimits_0^{10000} \frac{\epsilon^{(2r+1)} d\epsilon}{1+\exp(\epsilon - \eta)}$'
ke = r'$\displaystyle\kappa_e = L \sigma T = \frac{L T}{\rho}$'
thermal = r'$\displaystyle\kappa = D \cdot d \cdot c_p$'
kl = r'$\displaystyle \kappa_L = \kappa - \kappa_e$'
dulong = r'$\displaystyle c_p = \frac{3NR}{m}$'
mu0 = r'$\displaystyle \mu_0 = \mu_H \cdot \frac{(r+3/2)F_{(r+1/2)}}{(2r+3/2)F_{(2r+1/2)}}$'
e = r'$e = 1.60217653 \times 10^{-19}\, \mathrm{C}$'
kb = r'$k_B = 1.3806505 \times 10^{-23}\, \mathrm{J/K}$'
hbar = r'$\hbar = 1.054571817 \times 10^{-34}\, \mathrm{J \cdot s}$'
m = r'$m = 9.10938356 \times 10^{-31}\, \mathrm{kg}$'
planck = r'$\displaystyle\hbar = \frac{h}{2\pi}$'
lamb = r'$\displaystyle\lambda = r + 1/2$ in reference [1]'

equations = [psi, seebeck, seebeck_fermi, carrier, lorenz, efm, zt, beta, tzt, fermi, fermi_example, ke, thermal, kl, dulong, mu0, e, kb, hbar, m, planck, lamb]
equations_names = ['psi', 'seebeck', 'seebeck_fermi', 'carrier', 'lorenz', 'efm', 'zt', 'beta', 'tzt', 'fermi', 'fermi_example', 'ke', 'thermal', 'kl', 'dulong', 'mu0', 'e', 'kb', 'hbar', 'm', 'planck', 'lamb']

for eq in range(len(equations)):
	
	fig = plt.figure()
	text = fig.text(0, 0, equations[eq])

	# Saving the figure will render the text.
	dpi = 500
	fig.savefig(equations_names[eq], dpi=dpi)

	# Now we can work with text's bounding box.
	bbox = text.get_window_extent()
	width, height = bbox.size / float(dpi) + 0.005
	# Adjust the figure size so it can hold the entire text.
	fig.set_size_inches((width, height+0.05))

	# Adjust text's vertical position.
	dy = (bbox.ymin/float(dpi))/height
	text.set_position((0, -dy))
	# Save the adjusted text.
	fig.savefig(equations_names[eq] + '.png', dpi=dpi)

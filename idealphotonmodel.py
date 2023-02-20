import numpy as np
# import scipy as sc
from scipy.linalg import expm
import matplotlib.pyplot as plt

# operator for the quantum harmonic oscillator
i = 1j
Nmax = 7  # truncation to Nmax
Vacuum = np.zeros((Nmax + 1, 1))
Vacuum[0] = 1
Id = np.eye(Nmax + 1)
N = np.diag(np.arange(0, Nmax+1))  # photon number operator
A = np.diag(np.sqrt(np.arange(1, Nmax+1)), 1)  # annihilation operator

# measurement operators
ph0 = np.pi/8  # dephasing per photon
phR = np.pi/10  # phase of the second Ramsey pulse
Mg = np.diag(np.cos((ph0*np.arange(0, Nmax+1)+phR)/2))
Me = np.diag(np.sin((ph0*np.arange(0, Nmax+1)+phR)/2))

# simulation parameters
Niter = 300
buf = np.linalg.matrix_power(expm(np.sqrt(3)*(A-A.T)), 1) @ Vacuum
rho = buf @ buf.T.conj()
# rho = Id / np.trace(Id) # initial state (no information)

# simulation loop
Pop = np.zeros((Nmax+1, Niter))
y = np.zeros(Niter)
for ii in range(Niter):
    Pop[:, ii] = np.diag(rho)
    rhog = Mg @ rho @ Mg.T.conj()
    rhoe = Me @ rho @ Me.T.conj()
    if np.random.rand() < np.trace(rhog):
        y[ii] = -1
        rho = rhog / np.trace(rhog)
    else:
        y[ii] = 1
        rho = rhoe / np.trace(rhoe)

# graphics
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
axs[0].bar(np.arange(1, Niter+1), y, width=1.0)
axs[0].set_ylabel('Measurement outcome')
axs[0].set_ylim([-1.02, 1.02])
axs[1].plot(np.arange(1, Niter+1), Pop.T, linewidth=2)
axs[1].legend(np.arange(0, Nmax+1))
axs[1].set_ylim([-0.02, 1.02])
axs[1].set_ylabel('populations')
axs[1].set_xlabel('step number')
plt.show()

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# operator for the quantum harmonic oscillator
i = complex(0, 1)
Nmax = 10  # truncation to Nmax
Vacuum = np.zeros((Nmax+1, 1), dtype=complex)
Vacuum[0] = 1
Id = np.eye(Nmax+1, dtype=complex)
N = np.diag(np.arange(Nmax+1))  # photon number operator
A = np.diag(np.sqrt(np.arange(1, Nmax+1)), k=1)  # operator

# measurement operators
ph0 = np.pi/8  # dephasing per photon
phR = np.pi/10  # phase of the second Ramsey pulse
Mg = np.diag(np.cos((ph0*np.arange(Nmax+1)+phR)/2))
Me = np.diag(np.sin((ph0*np.arange(Nmax+1)+phR)/2))

# measurement imperfections
Eff = 0.8  # efficiency
Err = 0.2  # error rate

# cavity decoherence
nth = 0.1 # thermal photon
Tcav = 140e-3  # photon life time (s)
Tsampling = 80e-6  # sampling period (s)
Lm = np.sqrt((1+nth)/Tcav) * A.T.conj()  # Lindblad operator associated to photon lost
Lp = np.sqrt(nth/Tcav) * A  # Lindblad operator associated to photon gain
M0 = Id - Tsampling*(Lm.T @ Lm + Lp.T @ Lp)/2  # Kraus operator for zero photon annihilation
Mm = np.sqrt(Tsampling) * Lm  # Kraus operator to one photon annihilation
Mp = np.sqrt(Tsampling) * Lp  # Kraus operator to one photon creation

# simulation parameter
Niter = 5000
buf = expm(np.sqrt(3)*(A - A.T.conj())) @ Vacuum
rho = buf @ buf.T.conj()  # initial state (6 photons)

# simulation loop
Pop = np.zeros((Nmax+1, Niter))
y = np.zeros(Niter)
for i in range(Niter):
    Pop[:, i] = np.diag(rho)
    rhog0 = Mg @ rho @ Mg.T.conj()
    rhoe0 = Me @ rho @ Me.T.conj()
    rhog = (1 - Err) * rhog0 + Err * rhoe0
    rhoe = Err * rhog0 + (1 - Err) * rhoe0
    if np.random.rand() < Eff:
        if np.random.rand() < np.trace(rhog):
            y[i] = -1
            rho = rhog / np.trace(rhog)
        else:
            y[i] = 1
            rho = rhoe / np.trace(rhoe)
    else:
        y[i] = 0
        rho = rhog + rhoe
    rho = M0 @ rho @ M0.T.conj() + Mm @ rho @ Mm.T.conj() + Mp @ rho @ Mp.T.conj()
    rho = rho / np.trace(rho)

# graphics
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
ax1.bar(np.arange(1, Niter+1), y)
ax1.set_ylabel('Measurement outcome')
ax1.set_ylim([-1.02, 1.02])
ax2.plot(np.arange(1, Niter+1), Pop.T, linewidth=2)
ax2.legend(np.arange(Nmax+1))
ax2.set_ylabel('Populations')
ax2.set_xlabel('Step number')
ax2.set_ylim([-0.02, 1.02])
plt.show()

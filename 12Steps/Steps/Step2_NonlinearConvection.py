import numpy as np
from matplotlib import pyplot as plt


# equation to solve: du / dt + c * du / dx = 0 <-- hyperbolic transport equation
# note we use a noncentral differenzenquotienten!

# the amount of spacial points
nx = 41
nt = 20 # the amount of iterations

# the interval that will be looked at
x_min = 0
x_max = 2

# the length of each cell
dx = (x_max - x_min) / (nx - 1)

dt = 0.025
# wavespeed
c = 1

# initial condition for u: u = 2 for 0.5 <= x <= 1 else u = 1
u = np.ones(nx)

u[int((0.5 - x_min) / dx):int((1 - x_min) / dx + 1)] = 2

# show the initial conditions
# plt.plot(np.linspace(x_min, x_max, nx), u)
# plt.show()

# placeholder array for the new values
un = np.ones(nx)

for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - un[i] * dt * (un[i] - un[i - 1]) / dx

plt.plot(np.linspace(x_min, x_max, nx), u)
plt.show()


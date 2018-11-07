import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from Viewer import render_scale


nx = 81
ny = 81
nt = 500
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)


rho = 1
nu = .1
dt = .001


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))


### pygame shit
# the scale that will be rendered at. only integer multiples are possible
scale = 10
vec_scale = 100

# initiate pygame
pg.init()
screen = pg.display.set_mode((nx * scale, ny * scale))
pg.display.set_caption("Fire Animation")
clock = pg.time.Clock()
clock.tick()
font = pg.font.SysFont("comicsansms", 10)


# solves the poisson equation for every iteration
def pressure_poisson():
    global p
    b = np.zeros((ny, nx))
    b[1:-1, 1:-1] = dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * \
                    (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                             (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                         (2 * (dx ** 2 + dy ** 2)) - b[1:-1, 1:-1])

        # neumann randbedingungen
        p[:, -1] = p[:, -2]  ##dp/dy = 0 at x = 2
        p[0,:] = p[1,:]  ##dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]  ##dp/dx = 0 at x = 0
        # dirichtlet randbedingungen
        p[-1,:] = 0  ##p = 0 at y = 2


for n in range(nt):
    un = u.copy()
    vn = v.copy()

    pressure_poisson()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx ** 2 *
                           (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                           dt / dy ** 2 *
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx ** 2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                           dt / dy ** 2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = 1  # set velocity on cavity lid equal to 1
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

    render_scale(pg.surfarray.pixels3d(screen), p * 200, scale)
    # render velocity vectors
    for ii in range(int(v.shape[0] / 2)):
        i = ii * 2
        for jj in range(int(v.shape[1] / 2)):
            j = jj * 2
            pg.draw.line(screen, (0, 0, 0), (i * scale, j * scale), (i * scale + v[i, j] * vec_scale, j * scale + u[i, j] * vec_scale))

    pg.display.flip()


loop = True
while loop:
    clock.tick(10)
    for e in pg.event.get():
        if e.type == pg.KEYDOWN:
            loop = False
pg.quit()

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from Viewer import render_scale, legend


nx = 301
ny = 101
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)


rho = 1
nu = .1
dt = .00005


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

u[:, :] = 2
# v[50: 100, 50:100] = 1


### pygame shit
# the scale that will be rendered at. only integer multiples are possible
scale = 2
vec_scale = scale

# initiate pygame
pg.init()
screen = pg.display.set_mode((nx * scale, ny * scale))
pg.display.set_caption("Fire Animation")
clock = pg.time.Clock()
clock.tick()
font = pg.font.SysFont("comicsansms", 10)

# [y_start, y_end, x_start, x_end]
wall_mask = np.zeros(u.shape).astype(np.bool)


# solves the poisson equation for every iteration
def pressure_poisson():
    global p
    b = np.zeros((ny, nx))
    b[1:-1, 1:-1] = dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * \
                    rho * ((1 / dt *
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

        # dirichtlet randbedingungen
        p[:, -1] = 0
        p[:, 0] = 0

        # neumann (no pressure change on wall
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        # p[0,:] = p[1,:]  ##dp/dy = 0 at y = 0
        # neumann randbedingungen
        # p[-1,:] = p[-2, :]  ##p = 0 at y = 2
        # p[:, 0] = p[:, 1]
        # p[:, -1] = p[:, -2]

        # for w in walls:
            # pressure can't change through walls?
            # p[w[0]:w[1], w[2]] = p[w[0]:w[1], w[2] - 1]
            # p[w[0]:w[1], w[3]] = p[w[0]:w[1], w[3] + 1]
            # p[w[0], w[2] + 1:w[3] - 1] = p[w[0] - 1, w[2] + 1:w[3] - 1]
            # p[w[1], w[2] + 1:w[3] - 1] = p[w[1] + 1, w[2] + 1:w[3] - 1]

show_vec = False
lmb_down = False
rmb_down = False
loop = True
while loop:
    for e in pg.event.get():
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_ESCAPE:
                loop = False
            elif e.key == pg.K_v:
                show_vec = not show_vec
        elif e.type == pg.MOUSEBUTTONDOWN:
            if e.button == 1:
                lmb_down = True
            elif e.button == 3:
                rmb_down = True
        elif e.type == pg.MOUSEBUTTONUP:
            if e.button == 1:
                lmb_down = False
            elif e.button == 3:
                rmb_down = False
        elif e.type == pg.QUIT:
            loop = False

    if rmb_down:
        mx, my = pg.mouse.get_pos()
        mx = int(mx / scale)
        my = int(my / scale)

        wall_mask[my - 2:my + 2, mx - 2: mx + 2] = False
    if lmb_down:
        # draw wall
        mx, my = pg.mouse.get_pos()
        mx = int(mx / scale)
        my = int(my / scale)
        wall_mask[my - 2:my + 2, mx - 2: mx + 2] = True


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
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))) # + 0.001  ## cavity flow

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

    # in y direction
    u[:, 0] = 10
    u[:, -1] = u[:, -2]
    v[:, 0] = 0
    v[:, -1] = 0


    ## in x direction
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]
    v[0, :] = v[1, :]

    u[wall_mask] = 0
    v[wall_mask] = 0

    # for w in walls:
    #     v[w[0]:w[1], w[2]:w[3]] = 0
    #     u[w[0]:w[1], w[2]:w[3]] = 0
    render_scale(pg.surfarray.pixels3d(screen), np.sqrt(v ** 2 + u ** 2), scale, 0, 25, legend, wall_mask)
    # render_scale(pg.surfarray.pixels3d(screen), p, scale, -50, 50)
    if show_vec:
        # render velocity vectors
        for ii in range(int(v.shape[0] / 2)):
            i = ii * 2
            for jj in range(int(v.shape[1] / 2)):
                j = jj * 2
                pg.draw.line(screen, (0, 0, 0), (j * scale, i * scale),
                             (j * scale + u[i, j] * vec_scale, i * scale + v[i, j] * vec_scale))

    pg.display.flip()

pg.quit()

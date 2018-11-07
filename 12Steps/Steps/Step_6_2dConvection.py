import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from Viewer import render_scale

nx = 101
ny = 101
nt = 80
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx)) ##create a 1xn vector of 1's
v = np.ones((ny, nx))
un = np.ones((ny, nx))
vn = np.ones((ny, nx))

###Assign initial conditions

##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
# note that these are still only going into one direction. this will be unstable with negative initial velocities due to
# beeing onesided
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2


### pygame shit
# the scale that will be rendered at. only integer multiples are possible
scale = 6

# initiate pygame
pg.init()
screen = pg.display.set_mode((nx * scale, ny * scale))
pg.display.set_caption("Fire Animation")
clock = pg.time.Clock()
clock.tick()
font = pg.font.SysFont("comicsansms", 10)


for n in range(nt + 1): ##loop across number of time steps
    clock.tick(60)
    un = u.copy()
    vn = v.copy()

    u[1:, 1:] = un[1:, 1:] - (un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1])) - (vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    v[1:, 1:] = vn[1:, 1:] - (un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) - (vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]))

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    # render the amount of velocity
    render_scale(pg.surfarray.pixels3d(screen), np.sqrt(u[:, :] ** scale + v[:, :] ** scale) * 200, scale)

    # render velocity vectors
    for ii in range(int(v.shape[0] / 2)):
        i = ii * 2
        for jj in range(int(v.shape[1] / 2)):
            j = jj * 2
            pg.draw.line(screen, (0, 0, 0), (i * scale, j * scale), (i * scale + v[i, j] * 7, j * scale + u[i, j] * 7))

    pg.display.flip()


loop = True
while loop:
    clock.tick(10)
    for e in pg.event.get():
        if e.type == pg.KEYDOWN:
            loop = False
pg.quit()

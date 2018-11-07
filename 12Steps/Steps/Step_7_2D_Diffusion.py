import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from Viewer import render_scale

###variable declarations
nx = 31
ny = 31
nt = 200
nu = .05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))  # create a 1xn vector of 1's
un = np.ones((ny, nx))

###Assign initial conditions

##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
# note that these are still only going into one direction. this will be unstable with negative initial velocities due to
# beeing onesided
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

    u[1:-1, 1:-1] = un[1:-1, 1:-1] + nu * dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) + \
                    nu * dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    render_scale(pg.surfarray.pixels3d(screen), u * 400, scale)
    pg.display.flip()


loop = True
while loop:
    clock.tick(10)
    for e in pg.event.get():
        if e.type == pg.KEYDOWN:
            loop = False
pg.quit()

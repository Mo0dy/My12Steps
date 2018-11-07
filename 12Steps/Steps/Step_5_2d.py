import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from Viewer import render_scale

nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx)) ##create a 1xn vector of 1's
un = np.ones((ny, nx)) ##

###Assign initial conditions

##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2


### pygame shit
# the scale that will be rendered at. only integer multiples are possible
scale = 2

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
    # row, col = u.shape
    # for j in range(1, row):
    #     for i in range(1, col):
    #         u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
    #                               (c * dt / dy * (un[j, i] - un[j - 1, i])))
    #         u[0, :] = 1
    #         u[-1, :] = 1
    #         u[:, 0] = 1
    #         u[:, -1] = 1

    u[1:, 1:] = un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    render_scale(pg.surfarray.pixels3d(screen), u[:, :] * 400, scale)
    pg.display.flip()


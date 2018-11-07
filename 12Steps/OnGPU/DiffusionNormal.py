import numba as nb
import numpy as np
import pygame as pg
from numba import cuda

# the size of the simulated (uniform) grid
nx = 501
ny = 501

TPBx = 32
TPBy = 32
threadperblock = TPBx, TPBy
blockspergrid = (nx + threadperblock[0] - 1) // threadperblock[0], (ny + threadperblock[1] - 1) // threadperblock[1]


# the size of the grid cells if the simulated area goes from 0 to 2!
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
# the x and y values of each grid cell
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

dt = .00002  # the timestep taken each iteration
diff = 0.1  # the diffusion coefficient

u = np.zeros((ny, nx))  # the scalar field


@nb.guvectorize([(nb.uint8[:, :, :], nb.float64[:, :], nb.int8, nb.float64, nb.float64)], '(a,b,c),(e,f),(),(),()', target='parallel', cache=True)
def render_scale_bare(screen_mat, mat, s, min_val, max_val):
    delta = max_val - min_val
    scale = 255 / delta
    color = np.array([0, 0])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int((mat[i, j] - min_val) * scale)
            color[0] = 255
            color[1] = 255 - val
            color[2] = 255 - val
            for a in range(s):
                for b in range(s):
                    screen_mat[j * s + b, i * s + a,  0] = color[0]
                    screen_mat[j * s + b, i * s + a, 1] = color[1]
                    screen_mat[j * s + b, i * s + a, 2] = color[2]


@nb.guvectorize([(nb.float64[:, :], nb.float64[:, :])], '(a,b)->(a,b)', target="parallel", cache=True, fastmath=True)
def diffuse(l2, l):
    # l2 = l.copy()
    for ii in range(l.shape[1] - 2):
        i = ii + 1
        for jj in range(l.shape[0] - 2):
            j = jj + 1
            l[j, i] = l2[j, i] + diff * dt * (
                        (l2[j, i - 1] - 2 * l2[j, i] + l2[j, i + 1]) / dx ** 2 + (
                            l2[j - 1, i] - 2 * l2[j, i] + l2[j + 1, i]) / dy ** 2)
            #

@nb.guvectorize([(nb.float64[:, :], nb.float64[:, :])], '(a,b)->(a,b)', target="parallel", cache=True, fastmath=True)
def diffuse_ord3(l2, l):
    for ii in range(l.shape[1] - 6):
        i = ii + 3
        for jj in range(l.shape[0] - 6):
            j = jj + 3
            l[j, i] = l2[j, i] + diff * dt * (
                        (2 * l2[j, i - 3] - 27 * l2[j, i - 2] + 270 * l2[j, i - 1] - 490 * l2[j, i] + 270 * l2[j, i + 1] - 27 * l2[j, i + 2] + 2 * l2[j, i + 3]) / dx ** 2 / 180 +
                        (2 * l2[j - 3, i] - 27 * l2[j - 2, i] + 270 * l2[j - 1, i] - 490 * l2[j, i] + 270 * l2[j + 1, i] - 27 * l2[j + 2, i] + 2 * l2[j + 3, i]) / dy ** 2 / 180)


# @nb.guvectorize([(nb.float64[:, :], nb.float64[:, :])], '(a,b)->(a,b)', target="cuda")
@nb.cuda.jit
def diffuse_cuda(l2, l):
    sL = nb.cuda.shared.array(shape=(TPBx, TPBy), dtype=nb.float64)
    j, i = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bwx = cuda.blockDim.x
    bwy = cuda.blockDim.y

    if 0 < j < l.shape[0] - 1 and 0 < i < l.shape[1] - 1:
        sL[tx, ty] = l2[j, i]
        nb.cuda.syncthreads()

        if tx == 0:
            vl = l2[j - 1, i]
            vr = sL[tx + 1, ty]
        elif tx == bwx - 1:
            vl = sL[tx - 1, ty]
            vr = l2[j + 1, i]
        else:
            vr = sL[tx + 1, ty]
            vl = sL[tx - 1, ty]

        l[j, i] = sL[tx, ty] + diff * dt * (
                (l2[j, i - 1] - 2 * sL[tx, ty] + l2[j, i + 1]) / dx ** 2 + (
                vl - 2 * sL[tx, ty] + vr) / dy ** 2)

        nb.cuda.syncthreads()

# pygame stuff
# the scale that will be rendered at. only integer multiples are possible
scale = 1

# initiate pygame
pg.init()
screen = pg.display.set_mode((nx * scale, ny * scale))
pg.display.set_caption("Parallel Diffusion Equation Timestep: {}".format(dt))
clock = pg.time.Clock()
clock.tick()
font = pg.font.SysFont("comicsansms", 10)


lmb_down = False
rmb_down = False

# count iteration to toggle periodic events
iteration = 0

loop = True
while loop:
    clock.tick()
    # handle events. this should probably be converted to a data structure
    for e in pg.event.get():
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_ESCAPE:
                loop = False
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

    # get mouse position and convert to grid coordinates
    mx, my = pg.mouse.get_pos()
    mx = mx // scale
    my = my // scale

    if rmb_down:
        pass
    if lmb_down:
        u[my - 10:my + 10, mx - 10: mx + 10] = 10

    # update simulation
    diffuse(u.copy(), u)
    # diffuse_ord3(u.copy(), u)
    # d_arr = cuda.to_device(u.copy())
    # diffuse_cuda[blockspergrid, threadperblock](d_arr, u)


    # render
    if not iteration % 10:
        render_scale_bare(pg.surfarray.pixels3d(screen), u, scale, 0, 12)
        text = font.render("fps: {}".format(clock.get_fps()), False, (0, 0, 0))
        screen.blit(text, (0, 0))
        pg.display.flip()
    iteration += 1
pg.quit()

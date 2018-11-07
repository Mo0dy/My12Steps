import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from Viewer import render_scale, legend, render_scale_legend
import numba as nb
import scipy.ndimage
import time


nx = 201
ny = 201
nit = 30
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)


rho = 1
nu = .1
dt = .00001


curr_toggle = False


u1 = np.zeros((ny, nx))
v1 = np.zeros((ny, nx))
p1 = np.zeros((ny, nx))
# a matrix used to calculate p
b = np.zeros((ny, nx))

l = np.zeros((ny, nx))

diff_l = 0.1 # the diffusion of l
fb = -0.045
T0 = 2 # ambient temperature
l[:, :] = T0

# vorticity confinement # not working yet
vc = 0 # 0.02

# v[50: 100, 50:100] = 1



### pygame shit
# the scale that will be rendered at. only integer multiples are possible
scale = 4
vec_scale = scale / 10

# initiate pygame
pg.init()
screen = pg.display.set_mode((nx * scale, ny * scale))
pg.display.set_caption("Fire Animation")
clock = pg.time.Clock()
clock.tick()
font = pg.font.SysFont("comicsansms", 10)

# [y_start, y_end, x_start, x_end]
wall_mask = np.zeros(u1.shape).astype(np.bool)


ignition_values = np.random.random_integers(2, 12, 7)
ignition_values = scipy.ndimage.zoom(ignition_values, 20, order=0)


@nb.guvectorize([(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :])], '(a,b),(a,b),(a,b),(a,b)', target='parallel', cache=True)
def pressure_poisson_nb(p1, b, u, v):
    # precalculate some values
    # a = dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * rho
    # by2dx = 1 / 2 / dx
    # by2dy = 1 / 2 / dy
    # bydt = 1 / dt

    # calculate b
    # for ii in range(b.shape[1] - 2):
    #     i = ii + 1
    #     for jj in range(b.shape[0] - 2):
    #         j = jj + 1
    #         u_central_x = (u[j, i + 1] - u[j, i - 1]) * by2dx
    #         v_central_y = v[j + 1, i] - v[j - 1, i] * by2dy
    #         b[j, i] = a * (bydt * (u_central_x + v_central_y) - u_central_x * u_central_x
    #                   - 2 * (u[j + 1, i] - u[j - 1, i]) * by2dy * (v[j, i + 1] - v[j, i - 1]) * by2dx
    #                   - v_central_y * v_central_y)

    b[1:-1, 1:-1] = dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * \
                    rho * ((1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                             (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

    # solve for p
    for q in range(nit):
        p2 = p1.copy()
        dysq = dy ** 2
        dxsq = dx ** 2
        t_c = 1 / 2 / (dysq + dxsq)

        for ii in range(p1.shape[1] - 2):
            i = ii + 1
            for jj in range(p1.shape[0] - 2):
                j = jj + 1
                p1[j, i] = ((p2[j, i + 1] + p2[j, i - 1]) * dysq + (p2[j + 1, i] + p2[j - 1, i]) * dxsq) * t_c - b[j, i]

        # dirichtlet randbedingungen
        p1[:, -1] = p1[:, -2]
        p1[:, 0] = p1[:, 1]

        # neumann (no pressure change on wall
        p1[0, :] = p1[1, :]
        p1[-1, :] = p1[-2, :]


@nb.guvectorize([(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.boolean[:, :], nb.float64[:, :])], '(a,b),(a,b),(a,b),(a,b),(a,b)', target='parallel', cache=True)
def iteration_nb_upwind(p, u1, v1, wall_mask, t):
    a = dt / dx
    b = dt / dy
    c = a / rho / 2
    d = dt / dx ** 2
    e = dt / dy ** 2
    f = b / rho / 2

    u2 = u1.copy()
    v2 = v1.copy()

    # the distance between two cells
    h = np.sqrt(dx ** 2 + dy ** 2)

    # the absolute vorticity at every point
    # omega = np.zeros(u2.shape)
    # omega[1:-1, 1:-1] = (v2[1:-1, 2:] - v2[1:-1, :-2]) / dx - (u2[2:, 1:-1] - u2[:-2, 1:-1])
    # abs_omega = np.abs(omega)

    for ii in range(p1.shape[1] - 2):
        i = ii + 1
        for jj in range(p1.shape[0] - 2):
            j = jj + 1
            if wall_mask[j, i]:
                u1[j, i] = 0
                v1[j, i] = 0
            else:
                if u2[j, i] > 0:
                    a2 = u2[j, i] * a * (u2[j, i] - u2[j, i - 1])
                    a3 = u2[j, i] * a * (v2[j, i] - v2[j, i - 1])
                else:
                    a2 = u2[j, i] * a * (u2[j, i + 1] - u2[j, i])
                    a3 = u2[j, i] * a * (v2[j, i + 1] - v2[j, i])
                if v2[j, i] > 0:
                    b2 = v2[j, i] * b * (u2[j, i] - u2[j - 1, i])
                    b3 = v2[j, i] * b * (v2[j, i] - v2[j - 1, i])
                else:
                    b2 = v2[j, i] * b * (u2[j + 1, i] - u2[j, i])
                    b3 = v2[j, i] * b * (v2[j + 1, i] - v2[j, i])

                # calculate the gradient of the absolute vorticity.
                # grad_x = (abs_omega[j, i + 1] - abs_omega[j, i - 1]) / dx
                # grad_y = (abs_omega[j + 1, i] - abs_omega[j - 1, i]) / dy

                # the length of the gradient
                # abs_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
                # if abs_grad > 0:
                #     N_x = grad_x / abs_grad
                #     N_y = grad_y / abs_grad
                # else:
                #     N_x = 0
                #     N_y = 0

                # f_vort_x = vc * h * (N_y * omega[j, i])
                # f_vort_y = vc * h * (-N_x * omega[j, i])


                u1[j, i] = u2[j, i] - a2 - b2 - \
                    c * (p[j, i + 1] - p[j, i - 1]) + nu * (d * (u2[j, i + 1] - 2 * u2[j, i] + u2[j, i - 1]) + e * (u2[j + 1, i] - 2 * u2[j, i] + u2[j - 1, i])) # + f_vort_x

                v1[j, i] = v2[j, i] - a3 - b3 - \
                    f * (p[j + 1, i] - p[j - 1, i]) + nu * (d * (v2[j, i + 1] - 2 * v2[j, i] + v2[j, i - 1]) + e * (v2[j + 1, i] - 2 * v2[j, i] + v2[j - 1, i])) + \
                    fb * (t[j,i] - T0) # + f_vort_y

    # in y direction
    u1[:, 0] = 100
    u1[:, -1] = u1[:, -2]
    v1[:, 0] = v1[:, 1]
    v1[:, -1] = v1[:, -2]

    ## in x direction
    u1[0, :] = u1[1, :]
    u1[-1, :] = u1[-2, :]
    v1[-1, :] = v1[-2, :]
    v1[0, :] = v1[1, :]


# upwind convection
@nb.guvectorize([(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :])], '(a,b),(a,b),(a,b)', target='parallel', cache=True)
def convect_diffuse_l(l, u1, v1):
    l2 = l.copy()
    for ii in range(p1.shape[1] - 2):
        i = ii + 1
        for jj in range(p1.shape[0] - 2):
            j = jj + 1
            if u1[j, i] > 0:
                a = u1[j,i] * (l2[j, i] - l2[j, i-1]) / dx
            else:
                a = u1[j,i] * (l2[j, i + 1] - l2[j, i]) / dx
            if v1[j, i] > 0:
                b = v1[j,i] * (l2[j, i] - l2[j - 1, i]) / dy
            else:
                b = v1[j,i] * (l2[j + 1, i] - l2[j, i]) / dy
            l[j, i] = l2[j, i] - dt * (a + b) + diff_l * dt * ((l[j, i - 1] - 2 * l[j, i] + l[j, i + 1]) / dx ** 2 + (l[j - 1, i] - 2 * l[j, i] + l[j + 1, i]) / dy ** 2 )

    # dirichtlet randbedingungen
    l[:, -1] = T0
    l[:, 0] = T0

    # neumann (no pressure change on wall
    l[0, :] = T0
    l[-1, :] = T0


show_vec = False
lmb_down = False
rmb_down = False
l_pressed = False
view_mode = True
iteration = 0
last_m_x = 0
last_m_y = 0


loop = True
while loop:
    # start_time = time.time()
    clock.tick()
    for e in pg.event.get():
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_ESCAPE:
                loop = False
            elif e.key == pg.K_v:
                show_vec = not show_vec
            elif e.key == pg.K_r:
                wall_mask[:, :] = False
            elif e.key == pg.K_l:
                l_pressed = True
            elif e.key == pg.K_t:
                view_mode = not view_mode
            elif e.key == pg.K_i:
                l[1:-1, 1:5] = 10
        elif e.type == pg.KEYUP:
            if e.key == pg.K_l:
                l_pressed = False
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

    mx, my = pg.mouse.get_pos()
    mx = int(mx / scale)
    my = int(my / scale)

    if rmb_down:
        wall_mask[my - 2:my + 2, mx - 2: mx + 2] = False
    if lmb_down:
        # draw wall
        v_x = last_m_x - mx
        v_y = last_m_y - my
        u1[my - 10:my + 10, mx - 10: mx +10] -= v_x * 40
        v1[my - 10:my + 10, mx - 10: mx + 10] -= v_y * 40
        l[my - 10:my + 10, mx - 10: mx + 10] = 10
    if l_pressed:
        wall_mask[my - 2:my + 2, mx - 2: mx + 2] = True

    last_m_x = mx
    last_m_y = my

    # start_pressure = time.time()
    pressure_poisson_nb(p1, b, u1, v1)
    # start_iter = time.time()
    iteration_nb_upwind(p1, u1, v1, wall_mask, l)

    # start_convect = time.time()
    # convect l
    convect_diffuse_l(l, u1, v1)
    # end_convect = time.time()

    if not iteration % 5:
        if view_mode:
            render_scale_legend(pg.surfarray.pixels3d(screen), np.sqrt(v1 ** 2 + u1 ** 2), scale, 0, 300, legend, wall_mask)
        else:
            render_scale(pg.surfarray.pixels3d(screen), l, scale, 0, 10, wall_mask)
        # render_scale(pg.surfarray.pixels3d(screen), p, scale, -50, 50)
        if show_vec:
            # render velocity vectors
            for ii in range(int(v1.shape[0] / 2)):
                i = ii * 2
                for jj in range(int(v1.shape[1] / 2)):
                    j = jj * 2
                    pg.draw.line(screen, (0, 0, 0), (j * scale, i * scale),
                                 (j * scale + u1[i, j] * vec_scale, i * scale + v1[i, j] * vec_scale))

        pg.display.flip()
    iteration += 1
    # loop_end = time.time()
    # l_time = loop_end - start_time
    # print("pressure: %0.2f " % ((start_iter - start_pressure) / l_time * 100) + "%")
    # print("iteration: %0.2f " % ((start_convect - start_iter) / l_time * 100) + "%")
    # print("convection: %0.2f " % ((end_convect - start_convect) / l_time * 100) + "%")
pg.quit()

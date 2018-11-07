import numpy as np
from Viewer import *
import numba as nb
import pygame as pg
import timeit


sor_param = 1.93


class Simulation(object):
    # faster math (numba) less accurate
    fastmath = True
    target = "parallel"

    # the size of the simulated (uniform) grid
    nx = 201
    ny = 201
    # the amount of iterations to solve the poisson pressure equation
    nit = 30

    # the size of the grid cells if the simulated area goes from 0 to 2!
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    # the x and y values of each grid cell
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    rho = 1  # the density of the fluid
    nu = .1  # the viscosity of the fluid
    dt = .000007  # the timestep taken each iteration
    diff_l = 0.1  # the diffusion coefficient of l
    fb = -0.045  # the buoyant force coefficient of l
    T0 = 2  # ambient temperature (concentration of l)

    u1 = np.zeros((ny, nx))  # the x velocity
    v1 = np.zeros((ny, nx))  # the y velocity
    p1 = np.zeros((ny, nx))  # the pressure (used to correct navier stokes with projected method)
    b = np.zeros((ny, nx))   # a help matrix used to calculate p

    scalar_fields = []
    # t = np.ones((ny, nx)) * T0  # the diffusion_convected fluid. also might have a buoyant force
    wall_mask = np.zeros(u1.shape).astype(np.bool)  # the position of added walls

    def __init__(self):
        self.scalar_fields += [np.ones((self.ny, self.nx)) * self.T0 for i in range(2)]
        self.pressure_poisson = self.make_pressure_poisson()
        self.iteration = self.make_iteration()
        self.convect_diffuse = self.make_convect_diffuse()

    def update(self):
        self.pressure_poisson(self.p1, self.b, self.u1, self.v1)
        self.iteration(self.p1, self.u1, self.v1, self.wall_mask) # , self.t)
        for s in self.scalar_fields:
            self.convect_diffuse(s, self.u1, self.v1)

    def make_pressure_poisson(self):
        dx = self.dx
        dy = self.dy
        rho = self.rho
        dt = self.dt
        nit = self.nit
        @nb.guvectorize([(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :])],
                        '(a,b),(a,b),(a,b),(a,b)', target=self.target, nopython=True, fastmath=self.fastmath)
        def pressure_poisson(p1, b, u, v):
            # precalculate some values
            # calculate b
            for ii in range(b.shape[1] - 2):
                i = ii + 1
                for jj in range(b.shape[0] - 2):
                    j = jj + 1

                    b[j, i] = dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * rho * \
                              (((u[j, i + 1] - u[j, i - 1]) / (2 * dx) + (v[j + 1, i] - v[j - 1, i]) / (2 * dy)) / dt
                               - ((u[j, i + 1] - u[j, i - 1]) / (2 * dx)) ** 2
                               - 2 * ((u[j + 1, i] - u[j - 1, i]) / (2 * dy) * (v[j, i + 1] - v[j, i - 1]) / (2 * dx))
                               - ((v[j + 1, i] - v[j - 1, i]) / (2 * dy)) ** 2)

            # solve for p
            dysq = dy ** 2
            dxsq = dx ** 2
            t_c = 1 / 2 / (dysq + dxsq)
            p2 = p1.copy()

            # for q in range(nit):
            #     p2 = p1.copy()
            #
            #     for ii in range(p1.shape[1] - 2):
            #         i = ii + 1
            #         for jj in range(p1.shape[0] - 2):
            #             j = jj + 1
            #             p1[j, i] = ((p2[j, i + 1] + p2[j, i - 1]) * dysq + (p2[j + 1, i] + p2[j - 1, i]) * dxsq) * t_c - \
            #                        b[j, i]
            #
            #
            #     p1[:, -1] = p1[:, -2]
            #     p1[:, 0] = p1[:, 1]
            #     p1[0, :] = p1[1, :]
            #     p1[-1, :] = p1[-2, :]

            # solve for p
            for q in range(int(nit / 2)):
                for ii in range(p1.shape[1] - 2):
                    i = ii + 1
                    for jj in range(p1.shape[0] - 2):
                        j = jj + 1
                        p2[j, i] = ((p1[j, i + 1] + p2[j, i - 1]) * dysq + (p1[j + 1, i] + p2[j - 1, i]) * dxsq) * t_c - \
                                   b[j, i]

                p2[:, -1] = p2[:, -2]
                p2[:, 0] = p2[:, 1]
                p2[0, :] = p2[1, :]
                p2[-1, :] = p2[-2, :]

                p2[:, :] = (1 - sor_param) * p1 + sor_param * p2

                for ii in range(p1.shape[1] - 2):
                    i = ii + 1
                    for jj in range(p1.shape[0] - 2):
                        j = jj + 1
                        p1[j, i] = ((p2[j, i + 1] + p1[j, i - 1]) * dysq + (p2[j + 1, i] + p1[j - 1, i]) * dxsq) * t_c - \
                                   b[j, i]

                p1[:, -1] = p1[:, -2]
                p1[:, 0] = p1[:, 1]
                p1[0, :] = p1[1, :]
                p1[-1, :] = p1[-2, :]

                p1[:, :] = (1 - sor_param) * p2 + sor_param * p1
            #
        return pressure_poisson

    def make_iteration(self):
        dx = self.dx
        dy = self.dy
        rho = self.rho
        dt = self.dt
        nu = self.nu
        # fb = self.fb
        # T0 = self.T0

        @nb.guvectorize([(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.boolean[:, :])],
                        '(a,b),(a,b),(a,b),(a,b)', target=self.target, cache=True, fastmath=self.fastmath)
        def iteration_nb_upwind(p, u1, v1, wall_mask): #, t):
            a = dt / dx
            b = dt / dy
            c = a / rho / 2
            d = dt / dx ** 2
            e = dt / dy ** 2
            f = b / rho / 2

            u2 = u1.copy()
            v2 = v1.copy()

            # the distance between two cells
            # h = np.sqrt(dx ** 2 + dy ** 2)

            # the absolute vorticity at every point
            # omega = np.zeros(u2.shape)
            # omega[1:-1, 1:-1] = (v2[1:-1, 2:] - v2[1:-1, :-2]) / dx - (u2[2:, 1:-1] - u2[:-2, 1:-1])
            # abs_omega = np.abs(omega)

            for ii in range(p.shape[1] - 2):
                i = ii + 1
                for jj in range(p.shape[0] - 2):
                    j = jj + 1
                    if wall_mask[j, i]:
                        u1[j, i] = 0
                        v1[j, i] = 0
                    else:
                        if u2[j, i] > 0:
                            a2 = u2[j, i] * a * (u2[j, i] - u1[j, i - 1])
                            a3 = u2[j, i] * a * (v2[j, i] - v1[j, i - 1])
                        else:
                            a2 = u2[j, i] * a * (u2[j, i + 1] - u2[j, i])
                            a3 = u2[j, i] * a * (v2[j, i + 1] - v2[j, i])
                        if v2[j, i] > 0:
                            b2 = v2[j, i] * b * (u2[j, i] - u1[j - 1, i])
                            b3 = v2[j, i] * b * (v2[j, i] - v1[j - 1, i])
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
                                   c * (p[j, i + 1] - p[j, i - 1]) + nu * (
                                               d * (u2[j, i + 1] - 2 * u2[j, i] + u1[j, i - 1]) + e * (
                                                   u2[j + 1, i] - 2 * u2[j, i] + u1[j - 1, i]))  # + f_vort_x

                        v1[j, i] = v2[j, i] - a3 - b3 - \
                                   f * (p[j + 1, i] - p[j - 1, i]) + nu * (
                                               d * (v2[j, i + 1] - 2 * v2[j, i] + v1[j, i - 1]) + e * (
                                                   v2[j + 1, i] - 2 * v2[j, i] + v1[j - 1, i])) # + \
                                   # fb * (t[j, i] - T0)  # + f_vort_y

            # in y direction
            # u1[:, 0] = u1[:, 1]
            # u1[:, -1] = u1[:, -2]
            v1[:, 0] = v1[:, 1]
            v1[:, -1] = v1[:, -2]
            u1[:, 0] = 0
            u1[:, -1] = 0
            # v1[:, 0] = 0
            # v1[:, -1] = 0

            ## in x direction
            u1[0, :] = u1[1, :]
            u1[-1, :] = u1[-2, :]
            # v1[-1, :] = v1[-2, :]
            # v1[0, :] = v1[1, :]
            # u1[0, :] = 0
            # u1[-1, :] = 0
            v1[-1, :] = 0
            v1[0, :] = 0

            v1[:, :] = (1 - sor_param) * v2 + sor_param * v1
            u1[:, :] = (1 - sor_param) * u2 + sor_param * u1

        return iteration_nb_upwind

    def make_convect_diffuse(self):
        dx = self.dx
        dy = self.dy
        dt = self.dt
        T0 = self.T0
        diff_l = self.diff_l

        # upwind convection
        @nb.guvectorize([(nb.float64[:, :], nb.float64[:, :], nb.float64[:, :])], '(a,b),(a,b),(a,b)',
                        target=self.target, cache=True, fastmath=self.fastmath)
        def convect_diffuse(l, u1, v1):
            l2 = l.copy()
            for ii in range(l.shape[1] - 2):
                i = ii + 1
                for jj in range(l.shape[0] - 2):
                    j = jj + 1
                    if u1[j, i] > 0:
                        a = u1[j, i] * (l2[j, i] - l[j, i - 1]) / dx
                    else:
                        a = u1[j, i] * (l2[j, i + 1] - l2[j, i]) / dx
                    if v1[j, i] > 0:
                        b = v1[j, i] * (l2[j, i] - l[j - 1, i]) / dy
                    else:
                        b = v1[j, i] * (l2[j + 1, i] - l2[j, i]) / dy
                    l[j, i] = l2[j, i] - dt * (a + b) + diff_l * dt * (
                                (l[j, i - 1] - 2 * l2[j, i] + l2[j, i + 1]) / dx ** 2 + (
                                    l[j - 1, i] - 2 * l2[j, i] + l2[j + 1, i]) / dy ** 2)

            # dirichtlet randbedingungen
            l[:, -1] = T0
            l[:, 0] = T0

            # neumann (no pressure change on wall
            l[0, :] = T0
            l[-1, :] = T0

        return convect_diffuse

    def reset(self):
        self.u1 = np.zeros((self.ny, self.nx))  # the x velocity
        self.v1 = np.zeros((self.ny, self.nx))  # the y velocity
        self.p1 = np.zeros((self.ny, self.nx))  # the pressure (used to correct navier stokes with projected method)
        self.b = np.zeros((self.ny, self.nx))  # a help matrix used to calculate p

        self.scalar_fields =  [np.ones((self.ny, self.nx)) * self.T0 for i in range(2)]
        # t = np.ones((ny, nx)) * T0  # the diffusion_convected fluid. also might have a buoyant force
        self.wall_mask = np.zeros(self.u1.shape).astype(np.bool)  # the position of added walls


if __name__ == "__main__":
    sim = Simulation()
    # print(timeit.timeit("sim.update()", setup="from NavierStokes_CleanCode import Simulation; sim=Simulation()", number=300))

    ### pygame shit
    # the scale that will be rendered at. only integer multiples are possible
    scale = 4
    vec_scale = scale / 10

    # initiate pygame
    pg.init()
    screen = pg.display.set_mode((sim.nx * scale, sim.ny * scale))
    pg.display.set_caption("Fire Animation")
    clock = pg.time.Clock()
    clock.tick()
    font = pg.font.SysFont("comicsansms", 10)

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
                    sim.reset()
                elif e.key == pg.K_l:
                    l_pressed = True
                elif e.key == pg.K_t:
                    view_mode = not view_mode
                elif e.key == pg.K_i:
                    sim.scalar_fields[0][1:-1, 1:5] = 10
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
            # sim.wall_mask[my - 2:my + 2, mx - 2: mx + 2] = False
            v_x = last_m_x - mx
            v_y = last_m_y - my
            sim.u1[my - 10:my + 10, mx - 10: mx + 10] -= v_x * 40
            sim.v1[my - 10:my + 10, mx - 10: mx + 10] -= v_y * 40
            sim.scalar_fields[0][my - 10:my + 10, mx - 10: mx + 10] = 10
        if lmb_down:
            # draw wall
            v_x = last_m_x - mx
            v_y = last_m_y - my
            sim.u1[my - 10:my + 10, mx - 10: mx +10] -= v_x * 40
            sim.v1[my - 10:my + 10, mx - 10: mx + 10] -= v_y * 40
            sim.scalar_fields[1][my - 10:my + 10, mx - 10: mx + 10] = 10
        if l_pressed:
            sim.wall_mask[my - 2:my + 2, mx - 2: mx + 2] = True

        last_m_x = mx
        last_m_y = my

        sim.update()

        if not iteration % 5:
            if view_mode:
                render_scale_legend(pg.surfarray.pixels3d(screen), np.sqrt(sim.v1 ** 2 + sim.u1 ** 2), scale, 0, 300, legend, sim.wall_mask)
            else:
                # render_scale(pg.surfarray.pixels3d(screen), sim.t, scale, 2, 10, sim.wall_mask)
                render_scale2(pg.surfarray.pixels3d(screen), sim.scalar_fields[0], sim.scalar_fields[1], scale, 2, 10, sim.wall_mask)
            # render_scale(pg.surfarray.pixels3d(screen), p, scale, -50, 50)
            if show_vec:
                render_vectors(screen, sim.v1, sim.u1, scale, vec_scale)
            pg.display.flip()
        iteration += 1
    pg.quit()

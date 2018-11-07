import numba as nb
import numpy as np
import cv2 as cv
import pygame as pg


legend = cv.imread('Legend.png')
bbrad = cv.imread('BlackBodyRadiation.png')
bbrad = np.array(bbrad[0, :, :])
# cv.imshow('test2', np.tile(bbrad, (50, 1, 1)))

legend = np.array(legend[:, 0, :])
# cv.imshow('legend', np.tile(legend, (20, 1, 1)))
# cv.waitKey(0)


@nb.guvectorize([(nb.uint8[:, :, :], nb.float64[:, :], nb.int8, nb.float64, nb.float64, nb.int64[:, :], nb.boolean[:, :])], '(a,b,c),(e,f),(),(),(),(g,h),(e,f)', target='parallel', cache=True)
def render_scale_legend(screen_mat, mat, s, min_val, max_val, l, wall_mask):
    delta = max_val - min_val
    scale = l.shape[0] / delta
    color = np.array([0, 0])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            index = int((mat[i, j] - min_val) * scale)
            color[0] = l[index, 0]
            color[1] = l[index, 1]
            color[2] = l[index, 2]
            if wall_mask[i, j]:
                color[0] = 0
                color[1] = 0
                color[2] = 0
            for a in range(s):
                for b in range(s):
                    screen_mat[j * s + b, i * s + a,  0] = color[0]
                    screen_mat[j * s + b, i * s + a, 1] = color[1]
                    screen_mat[j * s + b, i * s + a, 2] = color[2]


@nb.guvectorize([(nb.uint8[:, :, :], nb.float64[:, :], nb.int8, nb.float64, nb.float64, nb.boolean[:, :])], '(a,b,c),(e,f),(),(),(),(e,f)', target='parallel', cache=True)
def render_scale(screen_mat, mat, s, min_val, max_val, wall_mask):
    delta = max_val - min_val
    scale = 255 / delta
    color = np.array([0, 0])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int((mat[i, j] - min_val) * scale)
            color[0] = 255
            color[1] = 255 - val
            color[2] = 255 - val
            if wall_mask[i, j]:
                color[0] = 0
                color[1] = 0
                color[2] = 0
            for a in range(s):
                for b in range(s):
                    screen_mat[j * s + b, i * s + a,  0] = color[0]
                    screen_mat[j * s + b, i * s + a, 1] = color[1]
                    screen_mat[j * s + b, i * s + a, 2] = color[2]


@nb.guvectorize([(nb.uint8[:, :, :], nb.float64[:, :], nb.float64[:, :] , nb.int8, nb.float64, nb.float64, nb.boolean[:, :])], '(a,b,c),(e,f),(e,f),(),(),(),(e,f)', target='parallel', cache=True)
def render_scale2(screen_mat, s1, s2, s, min_val, max_val, wall_mask):
    delta = max_val - min_val
    scale = 255 / delta
    color = np.array([0, 0])
    for i in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            val1 = int((s1[i, j] - min_val) * scale)
            val2 = int((s2[i, j] - min_val) * scale)
            color[0] = val1
            color[1] = 0
            color[2] = val2
            if wall_mask[i, j]:
                color[0] = 0
                color[1] = 0
                color[2] = 0
            for a in range(s):
                for b in range(s):
                    screen_mat[j * s + b, i * s + a,  0] = color[0]
                    screen_mat[j * s + b, i * s + a, 1] = color[1]
                    screen_mat[j * s + b, i * s + a, 2] = color[2]


@nb.guvectorize([(nb.uint8[:, :, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.int8, nb.float64, nb.float64, nb.boolean[:, :])], '(a,b,c),(e,f),(e,f),(e,f),(),(),(),(e,f)', target='parallel', cache=True)
def fire_viewer(screen_mat, s1, s2, s3, s, min_val, max_val, wall_mask):
    delta = max_val - min_val
    scale = 255 / delta
    bbrad_scale = bbrad.shape[0] / delta
    color = np.array([0, 0])
    for i in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            val1 = bbrad.shape[0] - int((s1[i, j] - min_val) * bbrad_scale)
            val2 = int(scale * s3[i, j])
            color[0] = bbrad[val1, 0] - val2
            color[1] = bbrad[val1, 1] - val2
            color[2] = bbrad[val1, 2] - val2
            if color[0] < 0:
                color[0] = 0
            if color[1] < 0:
                color[1] = 0
            if color[2] < 0:
                color[2] = 0

            if wall_mask[i, j]:
                color[0] = 0
                color[1] = 0
                color[2] = 0
            for a in range(s):
                for b in range(s):
                    screen_mat[j * s + b, i * s + a,  0] = color[0]
                    screen_mat[j * s + b, i * s + a, 1] = color[1]
                    screen_mat[j * s + b, i * s + a, 2] = color[2]


def render_vectors(screen, v1, u1, scale, vec_scale):
    # render velocity vectors
    for ii in range(int(v1.shape[0] / 2)):
        i = ii * 2
        for jj in range(int(v1.shape[1] / 2)):
            j = jj * 2
            pg.draw.line(screen, (0, 0, 0), (j * scale, i * scale),
                         (j * scale + u1[i, j] * vec_scale, i * scale + v1[i, j] * vec_scale))

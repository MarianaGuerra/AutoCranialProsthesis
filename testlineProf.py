# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import copy


def main():

    contour = np.array(np.loadtxt("contourimg05.txt", delimiter=' '))
    c = contour[:, 0:2]
    ini_falha = 200
    end_falha = 400
    # z_gs = np.polyfit(c[ini_falha:end_falha, 0], c[ini_falha:end_falha, 1], 4)
    # print("Coeficientes GS: " + str(z_gs))
    c_gap = np.concatenate((c[0:ini_falha, :], c[end_falha:c.shape[0], :]), axis=0)
    # fig, ax = plt.subplots()
    # ax.plot(c_gap[:, 0], c_gap[:, 1], 'mo', markersize=1)
    # ax.plot(c_gap[:, 0], -1*c_gap[:, 1], 'bo', markersize=1)
    # ax.set_xlim([0, 512])
    # ax.set_ylim([-512, 512])
    # plt.show()

    # c_falha_sample = []
    # sample = 100
    # for n in range(0, c_gap.shape[0]-sample, sample):
    #     c_falha_sample.append(c_gap[n])
    # c_falha_sample = np.asarray(c_falha_sample)
    # mean = np.mean(c_falha_sample, axis=0)

    xmin = np.amin(c_gap[:, 0])
    xmax = np.amax(c_gap[:, 0])
    x_mean_axis = (xmin + xmax)/2

    ymin = np.amin(c_gap[:, 1])
    ymax = np.amax(c_gap[:, 1])
    y_mean_axis = (ymin + ymax) / 2

    ref_point = np.asarray([x_mean_axis, y_mean_axis])
    ref_vector = np.array([1, 0])

    c_mirror = copy.deepcopy(c_gap)
    for m in range(0, c_mirror.shape[0]):
        point = c_mirror[m]
        point[1] = - point[1] + 2*ref_point[1]

    c_gap_aboveaxis = []
    c_gap_underaxis = []
    for n in range(0, c_gap.shape[0]):
        point = c_gap[n]
        if point[1] < ref_point[1]:
            c_gap_underaxis.append(point)
        else:
            c_gap_aboveaxis.append(point)
    c_gap_aboveaxis = np.asarray(c_gap_aboveaxis)
    c_gap_underaxis = np.asarray(c_gap_underaxis)

    seg_c_gs = []
    seg_c_mirror = []

    if c_gap_underaxis.shape[0] < c_gap_aboveaxis.shape[0]:
        seg_c_gap = c_gap_underaxis
        for n in range(0, c.shape[0]):
            point = c[n]
            if point[1] < ref_point[1]:
                seg_c_gs.append(point)
        for n in range(0, c_mirror.shape[0]):
            point = c_mirror[n]
            if point[1] < ref_point[1]:
                seg_c_mirror.append(point)
    else:
        seg_c_gap = c_gap_aboveaxis
        for n in range(0, c.shape[0]):
            point = c[n]
            if point[1] > ref_point[1]:
                seg_c_gs.append(point)
        for n in range(0, c_mirror.shape[0]):
            point = c_mirror[n]
            if point[1] > ref_point[1]:
                seg_c_mirror.append(point)

    seg_c_gs = np.asarray(seg_c_gs)
    seg_c_mirror = np.asarray(seg_c_mirror)
    seg_c_gap = np.asarray(seg_c_gap)

    # ptos = c_falha.shape[0]/2
    # c_falha_mirror = np.concatenate((c_falha[0:ptos, :], c_falha[0:ptos, :], c_mirror[(c_mirror.shape[0]/2):c_mirror.shape[0], :]), axis=0)
    # z = np.polyfit(c_falha_mirror[:, 0], c_falha_mirror[:, 1], 4)
    # print("Coeficientes: " + str(z))
    # # x = np.arange(c_falha[ptos, 0], c_falha[0, 0], 0.01)
    # x = np.arange(150, 450, 0.01)
    # est = np.polyval(z, x)
    # # est_gs = np.polyval(z_gs, x)

    fig, ax = plt.subplots()
    ax.plot(seg_c_gap[:, 0], seg_c_gap[:, 1], 'mo', markersize=1)
    ax.plot(seg_c_mirror[:, 0], seg_c_mirror[:, 1], 'bo', markersize=1)
    #ax.plot(seg_c_gs[:, 0], seg_c_gs[:, 1], 'go', markersize=1)
    far_point = ref_point + 300 * ref_vector
    mid_point = ref_point - 300 * ref_vector
    ax.plot([mid_point[0], far_point[0]], [mid_point[1], far_point[1]], 'r--')
    # ax.plot(c_falha_sample[:, 0], c_falha_sample[:, 1], 'gx', markersize=12)
    # ax.plot(c_falha[0:c_falha.shape[0]/2, 0], c_falha[0:c_falha.shape[0]/2, 1], 'bo', markersize=1)
    # ax.plot(c_mirror[(c_mirror.shape[0]/2):c_mirror.shape[0], 0], c_mirror[(c_mirror.shape[0]/2):c_mirror.shape[0], 1], 'go', markersize=1)
    # ax.plot(c_falha_mirror[:, 0], c_falha_mirror[:, 1], 'go', markersize=1)
    # ax.plot(x, est, 'ko', markersize=1)
    # ax.plot(x, est_gs, 'go', markersize=1)
    # ax.plot(c[150:450, 0], c[150:450, 1], 'mo', markersize=1)
    # ax.plot(c_mirror[700:c_mirror.shape[0], 0], c_mirror[700:c_mirror.shape[0], 1], 'bo', markersize=1)
    # ax.plot(c_falha[0:ptos, 0], c_falha[0:ptos, 1], 'go', markersize=1)
    # ax.plot(x, est, 'k-')
    # ax.plot(ref_point[0], ref_point[1], 'gx')
    # ax.plot(c_mirror[3, 0], c_mirror[3, 1], 'bx')
    # ax.plot(c_falha[3, 0], c_falha[3, 1], 'mx')
    # ax.plot(c10[1, 0], c10[1, 1], 'bx')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    plt.show()


if __name__ == '__main__':
    main()



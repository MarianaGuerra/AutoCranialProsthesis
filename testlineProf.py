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
    c_falha = np.concatenate((c[0:ini_falha, :], c[end_falha:c.shape[0], :]), axis=0)
    fig, ax = plt.subplots()
    ax.plot(c_falha[:, 0], c_falha[:, 1], 'mo', markersize=1)
    ax.plot(c_falha[:, 0], -1*c_falha[:, 1], 'bo', markersize=1)
    ax.set_xlim([0, 512])
    ax.set_ylim([-512, 512])
    plt.show()

    c_falha_sample = []
    sample = 20
    for n in range(0, c_falha.shape[0]-sample, sample):
        c_falha_sample.append(c_falha[n])
    c_falha_sample = np.asarray(c_falha_sample)
    mean = np.mean(c_falha_sample, axis=0)

    c_mirror = copy.deepcopy(c_falha)
    for m in range(0, c_mirror.shape[0]):
        point = c_mirror[m]
        point[1] = - point[1] + 2*mean[1]



    # ref_point = c[5]
    # ref_vector = np.array([1, 0])
    # c_mirror = []
    # sample = 1
    # for n in range(0, c.shape[0]-sample, sample):
    #     point = ref_point - (c[n] - ref_point) + 2 * ref_vector * np.dot((c[n] - ref_point), ref_vector)
    #     c_mirror.append(point)
    # c_mirror = np.asarray(c_mirror)

    # ptos = c_falha.shape[0]/2
    # c_falha_mirror = np.concatenate((c_falha[0:ptos, :], c_falha[0:ptos, :], c_mirror[(c_mirror.shape[0]/2):c_mirror.shape[0], :]), axis=0)
    # z = np.polyfit(c_falha_mirror[:, 0], c_falha_mirror[:, 1], 4)
    # print("Coeficientes: " + str(z))
    # #x = np.arange(c_falha[ptos, 0], c_falha[0, 0], 0.01)
    # x = np.arange(150, 450, 0.01)
    # est = np.polyval(z, x)
    # est_gs = np.polyval(z_gs, x)

    fig, ax = plt.subplots()
    ax.plot(c_falha[:, 0], c_falha[:, 1], 'mo', markersize=1)
    ax.plot(c_mirror[:, 0], c_mirror[:, 1], 'bo', markersize=1)
    # ax.plot(c_falha[0:c_falha.shape[0]/2, 0], c_falha[0:c_falha.shape[0]/2, 1], 'bo', markersize=1)
    # ax.plot(c_mirror[(c_mirror.shape[0]/2):c_mirror.shape[0], 0], c_mirror[(c_mirror.shape[0]/2):c_mirror.shape[0], 1], 'go', markersize=1)
    # ax.plot(c_falha_mirror[:, 0], c_falha_mirror[:, 1], 'go', markersize=1)
    # ax.plot(x, est, 'ko', markersize=1)
    # ax.plot(x, est_gs, 'go', markersize=1)
    # ax.plot(c[150:450, 0], c[150:450, 1], 'mo', markersize=1)
    # ax.plot(c_mirror[700:c_mirror.shape[0], 0], c_mirror[700:c_mirror.shape[0], 1], 'bo', markersize=1)
    # ax.plot(c_falha[0:ptos, 0], c_falha[0:ptos, 1], 'go', markersize=1)
    # ax.plot(x, est, 'k-')
    # ax.plot(c_mirror[400, 0], c_mirror[400, 1], 'gx')
    # ax.plot(c10[1, 0], c10[1, 1], 'bx')
    ax.set_xlim([0, 512])
    ax.set_ylim([-512, 512])
    plt.show()


if __name__ == '__main__':
    main()



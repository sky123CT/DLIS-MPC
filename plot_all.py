from mpc.functional.obstacle import Obstacle
from mpc.utility.utility import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    obstacle = Obstacle(num=1, shape=["circle"], center_list=[[-3.5, -0.25]], radius_list=[[1]], shrinking_par=0.03)
    plan_x, plan_y, plan_theta1, plan_theta2 = line_cross_2_obstacle()

    data_DL_RA10_NMPC_CBF = pd.read_csv("DL_RA10_NMPC_CBF.csv")
    data_DL_RA5_NMPC_CBF = pd.read_csv("DL_RA5_NMPC_CBF.csv")
    data_DL_RA1_NMPC_CBF = pd.read_csv("DL_RA1_NMPC_CBF.csv")
    tra1 = np.array(data_DL_RA10_NMPC_CBF.values)
    tra2 = np.array(data_DL_RA5_NMPC_CBF.values)
    tra3 = np.array(data_DL_RA1_NMPC_CBF.values)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(plan_x, plan_y, "-r", label="course")
    plt.plot(tra1[0, 1:], tra1[1, 1:], "-b", label="trajectory1")
    plt.plot(tra2[0, 1:], tra2[1, 1:], "-y", label="trajectory2")
    plt.plot(tra3[0, 1:], tra3[1, 1:], "-g", label="trajectory3")

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = obstacle.obstacle_list[0]["center"][0] + obstacle.obstacle_list[0]["radius"] * np.cos(theta)
    circle_y = obstacle.obstacle_list[0]["center"][1] + obstacle.obstacle_list[0]["radius"] * np.sin(theta)
    plt.plot(circle_x, circle_y)

    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

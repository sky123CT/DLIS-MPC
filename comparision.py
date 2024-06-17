from mpc.functional.dynamics import Dynamics
from mpc.utility.utility import *
from mpc.functional.car import Car
from mpc.functional.obstacle import Obstacle
from mpc.mpc_test import MPC
from dlis.model_learning.data.data_producer_1trailer import RandUP, PolygonOperator
from casadi import *
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import alphashape.alphashape as ConcaveHull
import time
import pandas as pd

CIR_RAD = 5  # radius of circular path [m]
TARGET_SPEED = 0.25  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number
MAX_STEP = 500  # max simulation time


NX = 4
NU = 2
DT = 0.1
T = 10
T_is = 5

def main():
    car = Car()
    obstacle = Obstacle(num=1, shape=["circle"], center_list=[[-3.5, 3]], radius_list=[[1]], shrinking_par=0.03)

    dynamics = Dynamics(car=car, nx=NX, nu=NU, dt=DT)
    # plan_x, plan_y, plan_theta1, ck = line_cross_2_obstacle()
    plan_x, plan_y, plan_theta1, ck = get_circle_course_backward(CIR_RAD)
    plan_theta2 = plan_theta1
    car.update_state(plan_x[0], plan_y[0], plan_theta1[0], plan_theta2[0])

    current_step = 0
    x = [car.state.x]
    y = [car.state.y]
    theta1 = [car.state.theta1]
    theta2 = [car.state.theta2]
    t = [0.0]
    dyaw = [0.0]
    v = [0.0]
    diff_tar_act = []
    target_ind, _ = calc_nearest_index(car.state, plan_x, plan_y, plan_theta2, 0, N_IND_SEARCH)

    x_list = [car.state.x + math.cos(car.state.theta2) * car.ROD_LEN]
    y_list = [car.state.y + math.sin(car.state.theta2) * car.ROD_LEN]

    plan_theta2 = smooth_yaw(plan_theta2)

    iteration_time = []
    while MAX_STEP >= current_step:
        start = time.time()
        x_ref, target_ind = calc_ref_trajectory(car.state,
                                                plan_x, plan_y, plan_theta1, plan_theta2,
                                                target_ind, NX, T, N_IND_SEARCH)
        x0 = [car.state.x, car.state.y, car.state.theta1, car.state.theta2]  # current state
        diff_tar_act.append([abs(x_ref[0, 0] - car.state.x), abs(x_ref[1, 0] - car.state.y)])

        new_mpc = MPC(car=car,
                      obstacle=obstacle,
                      dynamics=dynamics,
                      ref_state=x_ref,
                      init_state=x0,
                      nx=NX,
                      nu=NU,
                      dt=DT,
                      horizon=T,
                      horizon_is=T_is,
                      cbf_slack_activate=False)

        x1_opt, x2_opt, x3_opt, x4_opt, u1_opt, u2_opt = new_mpc.one_iter_mpc_control()
        del new_mpc

        taken_input = [u1_opt[0], u2_opt[0]]
        new_states = np.array(
            dynamics.dynamics_mapping(x0=DM([car.state.x, car.state.y, car.state.theta1, car.state.theta2]),
                                      p=DM(taken_input))['xf']
        ).squeeze()
        car.update_state(new_states[0], new_states[1], new_states[2], new_states[3])
        end = time.time()
        iteration_time.append((end - start)*1000)
        print('iteration time: %sms' % iteration_time[current_step])
        current_step += 1


        t4_states = [x1_opt[4], x2_opt[4], x3_opt[4], x4_opt[4]]

        x.append(car.state.x)
        y.append(car.state.y)
        theta1.append(car.state.theta1)
        theta2.append(car.state.theta2)

        x_list.append(car.state.x + np.cos(car.state.theta2) * car.ROD_LEN)
        y_list.append(car.state.y + np.sin(car.state.theta2) * car.ROD_LEN)

        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.subplot(211)
        if x1_opt is not None:
            plt.plot(x1_opt, x2_opt, "xr", label="MPC")
        plt.plot(plan_x, plan_y, "-r", label="course")
        plt.plot(x, y, "ob", label="trajectory")
        plt.plot(x_ref[0, :], x_ref[1, :], "xk", label="state_reference")
        plt.plot(plan_x[target_ind], plan_y[target_ind], "xg", label="target")

        car.plot_car(car.state.x, car.state.y, car.state.theta2, car.LENGTH_T)

        # for tractor
        car.plot_car(car.state.x + np.cos(car.state.theta2) * car.ROD_LEN + np.cos(car.state.theta1) * car.CP_OFFSET,
                     car.state.y + np.sin(car.state.theta2) * car.ROD_LEN + np.sin(car.state.theta1) * car.CP_OFFSET,
                     car.state.theta1,
                     car.LENGTH)
        plt.plot([car.state.x, car.state.x + np.cos(car.state.theta2) * car.ROD_LEN],
                 [car.state.y, car.state.y + np.sin(car.state.theta2) * car.ROD_LEN],
                 color='black', linewidth=2, linestyle='-')
        plt.plot([car.state.x + np.cos(car.state.theta2) * car.ROD_LEN,
                  car.state.x + np.cos(car.state.theta2) * car.ROD_LEN + np.cos(car.state.theta1) * car.CP_OFFSET],
                 [car.state.y + np.sin(car.state.theta2) * car.ROD_LEN,
                  car.state.y + np.sin(car.state.theta2) * car.ROD_LEN + np.sin(car.state.theta1) * car.CP_OFFSET],
                 color='black', linewidth=2, linestyle='--')
        x_list.append(car.state.x + np.cos(car.state.theta2) * car.ROD_LEN)
        y_list.append(car.state.y + np.sin(car.state.theta2) * car.ROD_LEN)

        for i in range(len(obstacle.obstacle_list)):
            if obstacle.obstacle_list[i]["shape"] == "polygon":
                # obstacle_x = np.concatenate((obstacle.vertices[:, 0], obstacle.vertices[0, 0]))
                # obstacle_y = np.concatenate((obstacle.vertices[:, 1], obstacle.vertices[0, 1]))
                # plt.plot(obstacle_x, obstacle_y)
                pass
            # plt.plot(txm, tym, "-y", label="tracking")
            elif obstacle.obstacle_list[i]["shape"] == "circle":
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = obstacle.obstacle_list[i]["center"][0] + obstacle.obstacle_list[i]["radius"] * np.cos(theta)
                circle_y = obstacle.obstacle_list[i]["center"][1] + obstacle.obstacle_list[i]["radius"] * np.sin(theta)
                plt.plot(circle_x, circle_y)

        plt.axis("equal")
        plt.grid(True)

        # plot reachable set and expanded obstacle
        plt.subplot(223)
        sampling_rs = RandUP(state=x0, real_time=True)
        polygon_operator = PolygonOperator(method_num=1, alpha=0.1)
        reachable_sets, _, _, sampling_rs_points, _ = RandUP.randup_trailer(vehicle=car,
                                                                            randup_process=sampling_rs,
                                                                            polygon_operator=polygon_operator)
        for i in range(len(reachable_sets)):
            reachable_bound = np.array(reachable_sets[i].boundary.coords)
            # sub-figure configuration
            plt.plot(reachable_bound[:, 0], reachable_bound[:, 1], 'g')

        for i in range(len(reachable_sets)):
            reachable_bound = np.array(reachable_sets[i].boundary.coords)
            # sub-figure configuration
            plt.plot(sampling_rs_points[i][:, 0], sampling_rs_points[i][:, 1], '.', color='b')
            # plt.plot(reachable_bound[:, 0], reachable_bound[:, 1], 'g')

        expanded_radius = obstacle.cbf_calculate_obstacle_expansion(
            robot_position=np.array([t4_states[0], t4_states[1]]))
        for i in range(len(obstacle.obstacle_list)):
            if obstacle.obstacle_list[i]["shape"] == "polygon":
                # obstacle_x = np.concatenate((obstacle.vertices[:, 0], obstacle.vertices[0, 0]))
                # obstacle_y = np.concatenate((obstacle.vertices[:, 1], obstacle.vertices[0, 1]))
                # plt.plot(obstacle_x, obstacle_y)
                pass
            # plt.plot(txm, tym, "-y", label="tracking")
            elif obstacle.obstacle_list[i]["shape"] == "circle":
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = (obstacle.obstacle_list[i]["center"][0] + obstacle.obstacle_list[i]["radius"] * np.cos(
                    theta)
                            + (expanded_radius[i] - obstacle.obstacle_list[i]["radius"]) * np.cos(theta) * (
                                    1 - obstacle.lam))
                circle_y = (obstacle.obstacle_list[i]["center"][1] + obstacle.obstacle_list[i]["radius"] * np.sin(
                    theta)
                            + (expanded_radius[i] - obstacle.obstacle_list[i]["radius"]) * np.sin(theta) * (
                                    1 - obstacle.lam))
                plt.plot(circle_x, circle_y)

        plt.xlim((car.state.x - 0.1, car.state.x + 0.1))
        plt.ylim((car.state.y - 0.1, car.state.y + 0.1))
        plt.grid(True)

        plt.subplot(224)

        angle_opt = -(x3_opt[5] - x4_opt[5])
        x1_ab = tan(
            (cos(sin(angle_opt) + (cos(angle_opt / -0.20138893) * -0.09728945))) ** 3 * -0.09880204)
        # ((math.cos(math.sin(x[i]) + (math.cos((0.5052427 / x[i]) + -0.3219844) * 0.13560449)))**3 * -0.10234546)
        y1_ab = (sin(sin(sin(angle_opt + (angle_opt - 0.029763347)))) * (
                -0.21348037 / ((sin(angle_opt / 0.51343226)) ** 3 + 3.8204532)))

        x2_ab = tan(tan(
            (cos(sin((angle_opt - (
                    cos(angle_opt * 1.1766043) * angle_opt) ** 2) + 0.09091717))) ** 2) * -0.06572991)
        y2_ab = sin(sin(sin(sin(angle_opt))) * -0.116112776)
        # ((math.sin((x[ind, 3] - (-0.013659489)**2) / 0.46065488) * -0.052274257) / math.cos(math.sin(x[ind, 3])))
        # (math.sin(x[ind, 3] / 0.6428589) * -0.07370442)

        x4_ab = ((cos(angle_opt) + -0.3333654) * -0.16466537)
        y4_ab = (angle_opt / ((-5.2116513 / cos(
            tan(sin(angle_opt)))) - 3.598078))

        rel_x1 = x1_ab * cos(x3_opt[5]) + y1_ab * -sin(
            x3_opt[5])  # @ [[cos(x[2, t]), -sin(x[2, t])],[sin(x[2, t]), cos(x[2, t])]]
        rel_y1 = x1_ab * sin(x3_opt[5]) + y1_ab * cos(x3_opt[5])
        x1 = x1_opt[5] + 2 * rel_x1
        y1 = x2_opt[5] + 2 * rel_y1

        rel_x2 = x2_ab * cos(x3_opt[5]) + y2_ab * -sin(x3_opt[5])
        rel_y2 = x2_ab * sin(x3_opt[5]) + y2_ab * cos(x3_opt[5])
        x2 = x1_opt[5] + 2 * rel_x2
        y2 = x2_opt[5] + 2 * rel_y2

        rel_x4 = x4_ab * cos(x3_opt[5]) + y4_ab * -sin(x3_opt[5])
        rel_y4 = x4_ab * sin(x3_opt[5]) + y4_ab * cos(x3_opt[5])
        x4 = x1_opt[5] + 2 * rel_x4
        y4 = x2_opt[5] + 2 * rel_y4

        ra_reachable_set = np.array([[x1, y1], [x4, y4], [x2, y2], [x1_opt[0], x2_opt[0]], [x1, y1]])
        plt.plot(ra_reachable_set[:, 0], ra_reachable_set[:, 1], 'g')

        expanded_radius = obstacle.cbf_calculate_obstacle_expansion(
            robot_position=np.array([t4_states[0], t4_states[1]]))
        for i in range(len(obstacle.obstacle_list)):
            if obstacle.obstacle_list[i]["shape"] == "polygon":
                # obstacle_x = np.concatenate((obstacle.vertices[:, 0], obstacle.vertices[0, 0]))
                # obstacle_y = np.concatenate((obstacle.vertices[:, 1], obstacle.vertices[0, 1]))
                # plt.plot(obstacle_x, obstacle_y)
                pass
            # plt.plot(txm, tym, "-y", label="tracking")
            elif obstacle.obstacle_list[i]["shape"] == "circle":
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = (obstacle.obstacle_list[i]["center"][0] + obstacle.obstacle_list[i]["radius"] * np.cos(
                    theta)
                            + (expanded_radius[i] - obstacle.obstacle_list[i]["radius"]) * np.cos(theta) * (
                                    1 - obstacle.lam))
                circle_y = (obstacle.obstacle_list[i]["center"][1] + obstacle.obstacle_list[i]["radius"] * np.sin(
                    theta)
                            + (expanded_radius[i] - obstacle.obstacle_list[i]["radius"]) * np.sin(theta) * (
                                    1 - obstacle.lam))
                plt.plot(circle_x, circle_y)

        plt.xlim((car.state.x - 0.5, car.state.x + 0.5))
        plt.ylim((car.state.y - 0.5, car.state.y + 0.5))
        plt.grid(True)


        """
        reachable_set_approx = [[car.state.x, car.state.y],
                                [car.state.x - 0.075 * 0.85 * np.cos(car.state.theta2) + 0.0025 * np.cos(
                                    car.state.theta2 + pi / 2),
                                 car.state.y - 0.075 * 0.85 * np.sin(car.state.theta2) + 0.0025 * np.sin(
                                     car.state.theta2 + pi / 2)],
                                [car.state.x - 0.075 * np.cos(car.state.theta2),
                                 car.state.y - 0.075 * np.sin(car.state.theta2)],
                                [car.state.x - 0.075 * 0.85 * np.cos(car.state.theta2) + 0.0025 * np.cos(
                                    car.state.theta2 - pi / 2),
                                 car.state.y - 0.075 * 0.85 * np.sin(car.state.theta2) + 0.0025 * np.sin(
                                     car.state.theta2 - pi / 2)],
                                [car.state.x, car.state.y]]
        reachable_x = []
        reachable_y = []
        for i in range(len(reachable_set_approx)):
            reachable_x.append(reachable_set_approx[i][0])
            reachable_y.append(reachable_set_approx[i][1])
        plt.plot(reachable_x, reachable_y, color='g')

        expanded_radius = obstacle.cbf_calculate_obstacle_expansion(
            robot_position=np.array([t4_states[0], t4_states[1]]))
        for i in range(len(obstacle.obstacle_list)):
            if obstacle.obstacle_list[i]["shape"] == "polygon":
                # obstacle_x = np.concatenate((obstacle.vertices[:, 0], obstacle.vertices[0, 0]))
                # obstacle_y = np.concatenate((obstacle.vertices[:, 1], obstacle.vertices[0, 1]))
                # plt.plot(obstacle_x, obstacle_y)
                pass
            # plt.plot(txm, tym, "-y", label="tracking")
            elif obstacle.obstacle_list[i]["shape"] == "circle":
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = (obstacle.obstacle_list[i]["center"][0] + obstacle.obstacle_list[i]["radius"] * np.cos(
                    theta)
                            + (expanded_radius[i] - obstacle.obstacle_list[i]["radius"]) * np.cos(theta) * (
                                    1 - obstacle.lam))
                circle_y = (obstacle.obstacle_list[i]["center"][1] + obstacle.obstacle_list[i]["radius"] * np.sin(
                    theta)
                            + (expanded_radius[i] - obstacle.obstacle_list[i]["radius"]) * np.sin(theta) * (
                                    1 - obstacle.lam))
                plt.plot(circle_x, circle_y)

        plt.xlim((car.state.x - 0.1, car.state.x + 0.1))
        plt.ylim((car.state.y - 0.1, car.state.y + 0.1))
        plt.grid(True)
        """
        plt.pause(0.0001)

    #result = [x, y]
    #pd.DataFrame(result).to_csv("DL_RA1_NMPC_CBF.csv")

if __name__ == "__main__":
    main()


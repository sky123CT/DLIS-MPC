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


CIR_RAD = 5  # radius of circular path [m]
TARGET_SPEED = 0.25  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number
MAX_TIME = 50.0  # max simulation time


NX = 4
NU = 2
DT = 0.1
T = 10
T_is = 5


def main():
    car = Car()
    obstacle = Obstacle(num=1, shape=["circle"], center_list=[[-3.5, -0.25]], radius_list=[[1]], shrinking_par=0.02)

    dynamics = Dynamics(car=car,
                        nx=NX,
                        nu=NU,
                        dt=DT)

    dl = 0.05  # course tick
    # cx, cy, cyawt, ck = get_circle_course_backward(CIR_RAD)
    cx, cy, cyawt, ck = line_cross_2_obstacle()
    sp = calc_speed_profile(cx, cy, cyawt, TARGET_SPEED)

    goal = [cx[-1], cy[-1]]
    car.update_state(cx[0], cy[0], cyawt[0], cyawt[0])

    Time = 0.0
    x = [car.state.x]
    y = [car.state.y]
    yaw = [car.state.theta1]
    yawt = [car.state.theta2]
    t = [0.0]
    dyaw = [0.0]
    v = [0.0]
    diff_tar_act = []
    target_ind, _ = calc_nearest_index(car.state, cx, cy, cyawt, 0, N_IND_SEARCH)

    txt_list = [car.state.x + math.cos(car.state.theta2) * car.ROD_LEN]
    tyt_list = [car.state.y + math.sin(car.state.theta2) * car.ROD_LEN]
    ov, odyaw = None, None

    cyawt = smooth_yaw(cyawt)

    start = time.time()

    while MAX_TIME >= Time:
        print(Time)
        xref, target_ind = calc_ref_trajectory(car.state, cx, cy, cyawt, cyawt, target_ind, NX, T, N_IND_SEARCH)
        x0 = [car.state.x, car.state.y, car.state.theta1, car.state.theta2]  # current state
        diff_tar_act.append([abs(xref[0, 0] - car.state.x), abs(xref[1, 0] - car.state.y)])

        new_mpc = MPC(car=car,
                      obstacle=obstacle,
                      dynamics=dynamics,
                      ref_state=xref,
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
        t4_states = [x1_opt[4], x2_opt[4], x3_opt[4], x4_opt[4]]

        Time = Time + DT

        x.append(car.state.x)
        y.append(car.state.y)
        yaw.append(car.state.theta1)
        yawt.append(car.state.theta2)
        t.append(Time)

        txt_list.append(car.state.x + np.cos(car.state.theta2) * car.ROD_LEN)
        tyt_list.append(car.state.y + np.sin(car.state.theta2) * car.ROD_LEN)

        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.subplot(311)
        if x1_opt is not None:
            plt.plot(x1_opt, x2_opt, "xr", label="MPC")
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(x, y, "ob", label="trajectory")
        plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
        plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        # TODO: the whole tractor-trailer system will be plot in one function
        car.plot_full_car()

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
        plt.subplot(312)
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

        plt.subplot(313)
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

        plt.pause(0.0001)

    diff = np.array(diff_tar_act)
    # print("error: ", sum([math.sqrt(i[0] ** 2 + i[1] ** 2) for i in diff]))

    res = [x, y]

    """
    print("min distance to obstacle: ", min(MIN_DIST))

    print("save data into csv!", res)
    pd.DataFrame(res).to_csv("res_cbf_02_20.csv")
    """

    if True:  # pragma: no cover
        # print("max iteration time", max(T_CAL))

        plt.close("all")
        figure, ax = plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="trailer")
        plt.plot(txt_list, tyt_list, "-b", label="tractor")
        if obstacle.shape == "polygon":
            obstacle_x = np.concatenate((obstacle.vertices[:, 0], obstacle.vertices[0, 0]))
            obstacle_y = np.concatenate((obstacle.vertices[:, 1], obstacle.vertices[0, 1]))
            plt.plot(obstacle_x, obstacle_y)
        # plt.plot(txm, tym, "-y", label="tracking")
        elif obstacle.shape == "circle":
            obstacle_circle = plt.Circle(obstacle.center_point, obstacle.radius, color="b", alpha=0.5)
            ax.add_patch(obstacle_circle)
        else:
            raise ValueError("Obstacle shape not given or not fit!")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.subplots()
        t = np.linspace(0, diff.shape[0], num=diff.shape[0])
        plt.plot(t, diff.T[0], "-g", label="x_axis")
        plt.plot(t, diff.T[1], "-b", label="y_axis")
        plt.plot(t, [math.sqrt(i[0] ** 2 + i[1] ** 2) for i in diff], "-r", label="distance")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("difference [m]")

        plt.show()


if __name__ == "__main__":
    main()



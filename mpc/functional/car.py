import numpy as np
import math
import matplotlib.pyplot as plt
from casadi import *


class State:
    """
    vehicle state class
    """

    def __init__(self, trailer_num=2, x=0.0, y=0.0, theta1=0.0, theta2=0.0, theta3=0.0):
        if trailer_num == 2:
            self.x = x
            self.y = y
            self.theta1 = theta1
            self.theta2 = theta2
            self.predelta = None
        elif trailer_num == 3:
            self.x = x
            self.y = y
            self.theta1 = theta1
            self.theta2 = theta2
            self.theta3 = theta3


class Car:
    def __init__(self, trailer_num=2, x=0.0, y=0.0, theta1=0.0, theta2=0.0, theta3=0.0):
        self.LENGTH = 0.72  # [m]
        self.LENGTH_T = 0.36  # [m]
        self.WIDTH = 0.48  # [m]
        self.BACKTOWHEEL = 0.36  # [m]
        self.WHEEL_LEN = 0.1  # [m]
        self.WHEEL_WIDTH = 0.05  # [m]
        self.TREAD = 0.2  # [m]
        self.WB = 0.3  # [m]
        self.ROD_LEN = 0.5  # [m]
        self.CP_OFFSET = 0.0  # [m]
        if trailer_num == 2:
            self.trailer_num = trailer_num
            self.state = State(trailer_num=trailer_num, x=x, y=y, theta1=theta1, theta2=theta2)

        elif trailer_num == 3:
            self.trailer_num = trailer_num
            self.state = State(trailer_num=trailer_num, x=x, y=y, theta1=theta1, theta2=theta2, theta3=theta3)

    def update_state(self, new_x=0.0, new_y=0.0, new_theta1=0.0, new_theta2=0.0, new_theta3=0.0):
        if self.trailer_num == 2:
            self.state.x = new_x
            self.state.y = new_y
            self.state.theta1 = new_theta1
            self.state.theta2 = new_theta2
        elif self.trailer_num == 3:
            self.state.x = new_x
            self.state.y = new_y
            self.state.theta1 = new_theta1
            self.state.theta2 = new_theta2
            self.state.theta3 = new_theta3

    def get_full_data(self):
        car_data = {}
        attach_points = {}
        if self.trailer_num == 2:
            center_trailer_x = self.state.x
            center_trailer_y = self.state.y
            orientation_trailer = self.state.theta2
            car_data['trailer'] = np.array([center_trailer_x, center_trailer_y, orientation_trailer])

            center_tractor_x = (self.state.x +
                                np.cos(self.state.theta2) * self.ROD_LEN +
                                np.cos(self.state.theta1) * self.CP_OFFSET)
            center_tractor_y = (self.state.y +
                                np.sin(self.state.theta2) * self.ROD_LEN +
                                np.sin(self.state.theta1) * self.CP_OFFSET)
            orientation_tractor = self.state.theta1
            car_data['tractor'] = np.array([center_tractor_x, center_tractor_y, orientation_tractor])

            attach_point_x = self.state.x + np.cos(self.state.theta2) * self.ROD_LEN
            attach_point_y = self.state.y + np.sin(self.state.theta2) * self.ROD_LEN
            attach_points['attach_point'] = np.array([attach_point_x, attach_point_y])

        elif self.trailer_num == 3:
            center_trailer2_x = self.state.x
            center_trailer2_y = self.state.y
            orientation_trailer2 = self.state.theta3
            car_data['trailer2'] = np.array([center_trailer2_x, center_trailer2_y, orientation_trailer2])

            attach_point2_x = self.state.x + np.cos(self.state.theta3) * self.ROD_LEN
            attach_point2_y = self.state.y + np.sin(self.state.theta3) * self.ROD_LEN
            attach_points['attach_point2'] = np.array([attach_point2_x, attach_point2_y])

            center_trailer1_x = (self.state.x +
                                 np.cos(self.state.theta3) * self.ROD_LEN +
                                 np.cos(self.state.theta2) * self.CP_OFFSET)
            center_trailer1_y = (self.state.y +
                                 np.sin(self.state.theta3) * self.ROD_LEN +
                                 np.sin(self.state.theta2) * self.CP_OFFSET)
            orientation_trailer1 = self.state.theta2
            car_data['trailer1'] = np.array([center_trailer1_x, center_trailer1_y, orientation_trailer1])

            attach_point1_x = (self.state.x +
                               np.cos(self.state.theta3) * self.ROD_LEN +
                               np.cos(self.state.theta2) * self.CP_OFFSET +
                               np.cos(self.state.theta2) * self.ROD_LEN)
            attach_point1_y = (self.state.y +
                                np.sin(self.state.theta3) * self.ROD_LEN +
                                np.sin(self.state.theta2) * self.CP_OFFSET +
                                np.sin(self.state.theta2) * self.ROD_LEN)
            attach_points['attach_point1'] = np.array([attach_point1_x, attach_point1_y])

            center_tractor_x = (self.state.x +
                                np.cos(self.state.theta3) * self.ROD_LEN +
                                np.cos(self.state.theta2) * self.CP_OFFSET +
                                np.cos(self.state.theta2) * self.ROD_LEN +
                                np.cos(self.state.theta1) * self.CP_OFFSET)
            center_tractor_y = (self.state.y +
                                np.sin(self.state.theta3) * self.ROD_LEN +
                                np.sin(self.state.theta2) * self.CP_OFFSET +
                                np.sin(self.state.theta2) * self.ROD_LEN +
                                np.sin(self.state.theta1) * self.CP_OFFSET)
            orientation_tractor = self.state.theta1
            car_data['tractor'] = np.array([center_tractor_x, center_tractor_y, orientation_tractor])

        return car_data, attach_points

    def get_full_data_symbolic(self, state_symbolic):
        car_data = {}
        attach_points = {}
        if self.trailer_num == 2:
            pass

        elif self.trailer_num == 3:
            center_trailer2_x = state_symbolic[0]
            center_trailer2_y = state_symbolic[1]
            orientation_trailer2 = state_symbolic[4]
            car_data['trailer2'] = vertcat(center_trailer2_x, center_trailer2_y, orientation_trailer2)

            attach_point2_x = state_symbolic[0] + np.cos(state_symbolic[4]) * self.ROD_LEN
            attach_point2_y = state_symbolic[1] + np.sin(state_symbolic[4]) * self.ROD_LEN
            attach_points['attach_point2'] = vertcat(attach_point2_x, attach_point2_y)

            center_trailer1_x = (state_symbolic[0] +
                                 np.cos(state_symbolic[4]) * self.ROD_LEN +
                                 np.cos(state_symbolic[3]) * self.CP_OFFSET)
            center_trailer1_y = (state_symbolic[1] +
                                 np.sin(state_symbolic[4]) * self.ROD_LEN +
                                 np.sin(state_symbolic[3]) * self.CP_OFFSET)
            orientation_trailer1 = state_symbolic[3]
            car_data['trailer1'] = vertcat(center_trailer1_x, center_trailer1_y, orientation_trailer1)

            attach_point1_x = (state_symbolic[0] +
                               np.cos(state_symbolic[4]) * self.ROD_LEN +
                               np.cos(state_symbolic[3]) * self.CP_OFFSET +
                               np.cos(state_symbolic[3]) * self.ROD_LEN)
            attach_point1_y = (state_symbolic[1] +
                               np.sin(state_symbolic[4]) * self.ROD_LEN +
                               np.sin(state_symbolic[3]) * self.CP_OFFSET +
                               np.sin(state_symbolic[3]) * self.ROD_LEN)
            attach_points['attach_point1'] = vertcat(attach_point1_x, attach_point1_y)

            center_tractor_x = (state_symbolic[0] +
                                np.cos(state_symbolic[4]) * self.ROD_LEN +
                                np.cos(state_symbolic[3]) * self.CP_OFFSET +
                                np.cos(state_symbolic[3]) * self.ROD_LEN +
                                np.cos(state_symbolic[2]) * self.CP_OFFSET)
            center_tractor_y = (state_symbolic[1] +
                                np.sin(state_symbolic[4]) * self.ROD_LEN +
                                np.sin(state_symbolic[3]) * self.CP_OFFSET +
                                np.sin(state_symbolic[3]) * self.ROD_LEN +
                                np.sin(state_symbolic[2]) * self.CP_OFFSET)
            orientation_tractor = state_symbolic[2]
            car_data['tractor'] = vertcat(center_tractor_x, center_tractor_y, orientation_tractor)

        return car_data, attach_points

    def plot_full_car(self):
        car_data, attach_points = self.get_full_data()
        if self.trailer_num == 2:
            center_trailer_x, center_trailer_y, orientation_trailer = car_data['trailer']
            center_tractor_x, center_tractor_y, orientation_tractor = car_data['tractor']
            attach_point_x, attach_point_y = attach_points['attach_point']
            # for trailer
            self.plot_car(center_trailer_x, center_trailer_y, orientation_trailer, self.LENGTH_T)
            # for tractor
            self.plot_car(center_tractor_x, center_tractor_y, orientation_tractor, self.LENGTH)
            plt.plot([center_trailer_x, attach_point_x], [center_trailer_y, attach_point_y],
                     color='black', linewidth=2, linestyle='-')
            plt.plot([attach_point_x, center_tractor_x], [attach_point_y, center_tractor_y],
                     color='black', linewidth=2, linestyle='--')

        elif self.trailer_num == 3:
            center_trailer2_x, center_trailer2_y, orientation_trailer2 = car_data['trailer2']
            center_trailer1_x, center_trailer1_y, orientation_trailer1 = car_data['trailer1']
            center_tractor_x, center_tractor_y, orientation_tractor = car_data['tractor']
            attach_point1_x, attach_point1_y = attach_points['attach_point1']
            attach_point2_x, attach_point2_y = attach_points['attach_point2']

            # for trailer2
            self.plot_car(center_trailer2_x, center_trailer2_y, orientation_trailer2, self.LENGTH_T)
            # for trailer1
            self.plot_car(center_trailer1_x, center_trailer1_y, orientation_trailer1, self.LENGTH_T)
            # for tractor
            self.plot_car(center_tractor_x, center_tractor_y, orientation_tractor, self.LENGTH)

            # for attachment2
            plt.plot([center_trailer2_x, attach_point2_x], [center_trailer2_y, attach_point2_y],
                     color='black', linewidth=2, linestyle='-')
            plt.plot([attach_point2_x, center_trailer1_x], [attach_point2_y, center_trailer1_y],
                     color='black', linewidth=2, linestyle='--')
            # for attachment2
            plt.plot([center_trailer1_x, attach_point1_x], [center_trailer1_y, attach_point1_y],
                     color='black', linewidth=2, linestyle='-')
            plt.plot([attach_point1_x, center_tractor_x], [attach_point1_y, center_tractor_y],
                     color='black', linewidth=2, linestyle='--')

        plt.gca().set_aspect('equal', adjustable='box')
        #plt.show()

    def plot_car(self, x, y, yaw, length, truck_color="-k"):  # pragma: no cover

        """
        Plot a car's position and orientation in a 2D plot.

        Args:
        x: x-coordinate of the car's center.
        y: y-coordinate of the car's center.
        yaw: yaw angle (orientation) of the car.
        length: length of the car.
        truck_color: color of the car's truck (default is black).
        """

        outline = np.array(
            [[-length / 2, (length - length / 2), (length - length / 2), -length / 2, -length / 2],
             [self.WIDTH / 2, self.WIDTH / 2, - self.WIDTH / 2, -self.WIDTH / 2, self.WIDTH / 2]]
        )

        rr_wheel = np.array(
            [[self.WHEEL_LEN, -self.WHEEL_LEN, -self.WHEEL_LEN, self.WHEEL_LEN, self.WHEEL_LEN],
             [-self.WHEEL_WIDTH - self.TREAD, -self.WHEEL_WIDTH - self.TREAD, self.WHEEL_WIDTH - self.TREAD,
              self.WHEEL_WIDTH - self.TREAD, -self.WHEEL_WIDTH - self.TREAD]]
        )

        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])

        outline = (outline.T.dot(rot1)).T
        rr_wheel = (rr_wheel.T.dot(rot1)).T
        rl_wheel = (rl_wheel.T.dot(rot1)).T

        outline[0, :] += x
        outline[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y

        # Plot the car's outline and wheels
        plt.plot(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), truck_color)
        plt.plot(np.array(rr_wheel[0, :]).flatten(), np.array(rr_wheel[1, :]).flatten(), truck_color)
        plt.plot(np.array(rl_wheel[0, :]).flatten(), np.array(rl_wheel[1, :]).flatten(), truck_color)
        plt.plot(x, y, "*")



import numpy as np
import math
import matplotlib.pyplot as plt


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, yawt=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.predelta = None


class Car:
    def __init__(self, state=State()):
        self.state = state

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

    def update_state(self, new_x, new_y, new_yaw, new_yawt):
        self.state.x = new_x
        self.state.y = new_y
        self.state.yaw = new_yaw
        self.state.yawt = new_yawt

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

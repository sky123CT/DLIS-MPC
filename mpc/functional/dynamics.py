from casadi import *
from mpc.functional.car import Car
import numpy as np
import matplotlib.pyplot as plt

class Dynamics:
    def __init__(self,
                 car: Car,
                 nx,
                 nu,
                 dt=0.1):
        names = self.__dict__
        self.NX = nx
        self.NU = nu
        self.car = car
        self.DT = dt

        self.x = []
        self.u = []
        for i in range(self.NX):
            names['x' + str(i)] = SX.sym('x' + str(i))
            self.x = vertcat(self.x, names['x' + str(i)])
        # define input
        for i in range(self.NU):
            names['u' + str(i)] = SX.sym('u' + str(i))
            self.u = vertcat(self.u, names['u' + str(i)])

        # define dynamics mapping
        self.dynamics_mapping = self.__define_dynamics_mapping()

    def __define_dynamics_mapping(self):
        if self.car.trailer_num == 2:
            x_dot_0 = self.u[0] * cos(self.x[2] - self.x[3]) * cos(self.x[3])
            x_dot_1 = self.u[0] * cos(self.x[2] - self.x[3]) * sin(self.x[3])
            x_dot_2 = self.u[1]
            x_dot_3 = (self.u[0] / self.car.ROD_LEN * sin(self.x[2] - self.x[3]) -
                       self.car.CP_OFFSET * self.u[0] * self.u[1] * cos(self.x[2] - self.x[3]) / self.car.ROD_LEN)
            x_dot = vertcat(x_dot_0, x_dot_1, x_dot_2, x_dot_3)

        elif self.car.trailer_num == 3:
            x_dot_0 = self.u[0] * cos(self.x[2] - self.x[3]) * cos(self.x[3] - self.x[4]) * cos(self.x[4])
            x_dot_1 = self.u[0] * cos(self.x[2] - self.x[3]) * cos(self.x[3] - self.x[4]) * sin(self.x[4])
            x_dot_2 = self.u[1]
            x_dot_3 = (self.u[0] / self.car.ROD_LEN * sin(self.x[2] - self.x[3]) -
                       self.car.CP_OFFSET * self.u[0] * self.u[1] * cos(self.x[2] - self.x[3]) / self.car.ROD_LEN)
            x_dot_4 = (self.u[0] * cos(self.x[2] - self.x[3]) / self.car.ROD_LEN * sin(self.x[3] - self.x[4]) -
                       self.car.CP_OFFSET * self.u[0] * cos(self.x[2] - self.x[3]) * x_dot_3 * cos(self.x[3] - self.x[4]) / self.car.ROD_LEN)
            x_dot = vertcat(x_dot_0, x_dot_1, x_dot_2, x_dot_3, x_dot_4)
        else:
            x_dot = []
            print('lack of trailer_num!')

        dae = {'x': self.x, 'p': self.u, 'ode': x_dot}
        opts = {'tf': self.DT}
        motion_prediction_mapping = integrator('F', 'cvodes', dae, opts)
        return motion_prediction_mapping


if __name__ == '__main__':
    car = Car(trailer_num=3)
    dynamics = Dynamics(car=car, nx=5, nu=2)
    step = 20
    u = np.ones((step, 2)) * 1
    u[:, 0] += np.ones(step)
    for i in range(step):
        new_states = np.array(
            dynamics.dynamics_mapping(
                x0=DM([car.state.x, car.state.y, car.state.theta1, car.state.theta2, car.state.theta3]),
                p=DM(u[i]))['xf']
        ).squeeze()
        plt.clf()
        car.update_state(new_states[0], new_states[1], new_states[2], new_states[3], new_states[4])
        car.plot_full_car()
        plt.pause(1)
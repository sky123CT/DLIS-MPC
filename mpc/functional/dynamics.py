from casadi import *
from .car import Car


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

        # define state
        for i in range(self.NX):
            names['x'+str(i)] = SX.sym('x'+str(i))
        self.x = vertcat(self.x0, self.x1, self.x2, self.x3)

        # define input
        for i in range(self.NU):
            names['u' + str(i)] = SX.sym('u' + str(i))
        self.u = vertcat(self.u0, self.u1)

        # define dynamics mapping
        self.dynamics_mapping = self.__define_dynamics_mapping()

    def __define_dynamics_mapping(self):
        x_dot_0 = self.u[0] * cos(self.x[2] - self.x[3]) * cos(self.x[3])
        x_dot_1 = self.u[0] * cos(self.x[2] - self.x[3]) * sin(self.x[3])
        x_dot_2 = self.u[1]
        x_dot_3 = (self.u[0] / self.car.ROD_LEN * sin(self.x[2] - self.x[3]) -
                   self.car.CP_OFFSET * self.u[1] * cos(self.x[2] - self.x[3]) / self.car.ROD_LEN)
        x_dot = vertcat(x_dot_0, x_dot_1, x_dot_2, x_dot_3)
        if False:
            x_next = self.x + x_dot * self.DT
            motion_prediction_mapping = Function('F', [self.x, self.u], [x_next], ['x0', 'p'], ['xf'])
        else:
            dae = {'x': self.x, 'p': self.u, 'ode': x_dot}
            opts = {'tf': self.DT}
            motion_prediction_mapping = integrator('F', 'cvodes', dae, opts)
        return motion_prediction_mapping


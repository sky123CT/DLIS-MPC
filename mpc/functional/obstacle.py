import numpy as np
from casadi import *
from ..utility.utility import *


class Obstacle:
    def __init__(self, shape="polygon", if_cbf=True):
        self.shape = shape
        if self.shape == "circle":
            self.center_point = np.array([-3, 4.2])
            self.radius = np.array([1.0])
            self.O_D = [100000000]
        elif self.shape == "polygon":
            self.center_point = np.array([-3, 4.2])
            self.vertices = np.array([[0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5]] + self.center_point)

        if if_cbf is True:
            self.lam = 0.05
            self.circle_robot_limit = 55

    def expand_polygon_as_circle(self, state_sym):
        center = self.center_point
        radius = np.linalg.norm(self.vertices-center, axis=-1)
        obstacle_radius = np.max(radius)
        i_vertices = np.argmax(radius)
        dis_rob2obs = sqrt((state_sym[0]-center[0]) ** 2 + (state_sym[1]-center[1]) ** 2)
        expand_rate = sqrt(dis_rob2obs ** 2 /
                           ((self.vertices[i_vertices]-center)[0]**2 +
                            (self.vertices[i_vertices]-center)[1]**2)
                           )
        expanded_vertices = mtimes(expand_rate, (self.vertices-center)) + repmat(center, 1, 4).T
        return expanded_vertices, expand_rate, obstacle_radius, i_vertices

    def distance_car2obstacle_sym(self, state_sym, t):
        distance = ((state_sym[0, t + 1] - self.center_point[0]) ** 2 +
                    (state_sym[1, t + 1] - self.center_point[1]) ** 2)
        return distance

    def cbf_sym(self, state_sym, t):
        cbf_sym = (self.distance_car2obstacle_sym(state_sym, t) -
                   (1 - self.lam) * self.distance_car2obstacle_sym(state_sym, t+1))
        return cbf_sym

    def cbf_calculate_obstacle_expansion(self, robot_position):
        if self.shape == "polygon":
            vertices_vectors = self.vertices - self.center_point
            robot_vector = robot_position - self.center_point
            robot_angle = np.arctan2(robot_vector[1], robot_vector[0])
            vertices_angles = np.arctan2(vertices_vectors[:, 1], vertices_vectors[:, 0])
            is_between = 0
            if robot_angle >= np.max(vertices_angles) or robot_angle <= np.min(vertices_angles):
                is_between = np.argmax(vertices_angles)
            else:
                for i in range(vertices_angles.shape[0]):
                    if vertices_angles[i] <= robot_angle <= vertices_angles[i - 1]:
                        is_between = i
            if is_between == vertices_vectors.shape[0]-1:
                expand_rate = Obstacle.cbf_expand_rate_2d(robot_vector,
                                                          np.vstack((vertices_vectors[is_between, :],
                                                                     vertices_vectors[0])))
            else:
                expand_rate = Obstacle.cbf_expand_rate_2d(robot_vector, vertices_vectors[is_between:is_between+2, :])
            expanded_vertices = vertices_vectors * expand_rate + self.center_point
            return expanded_vertices
        elif self.shape == "circle":
            expanded_radius = casadi_distance(self.center_point, robot_position)
            return expanded_radius
        else:
            raise ValueError("Obstacle shape not given or not fit!")


    @staticmethod
    def cbf_expand_rate_2d(robot_position, segment):
        return ((robot_position[0]*(segment[1, 1]-segment[0, 1])-robot_position[1]*(segment[1, 0]-segment[0, 0]))
                / (segment[0, 0]*segment[1, 1]-segment[0, 1]*segment[1, 0]))


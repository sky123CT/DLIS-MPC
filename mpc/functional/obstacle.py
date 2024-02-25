import numpy as np
from casadi import *
from ..utility.utility import *


class Obstacle:
    def __init__(self,
                 num=1,
                 shape=["polygon"],
                 center_list=[[-3.5, 2.5]],
                 radius_list=[[1.0]],
                 vertices_list=[[[0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5]]],
                 if_cbf=True):
        self.obs_num = num
        self.obstacle_list = []
        for i in range(self.obs_num):
            if shape[i] == "circle":
                self.obstacle_list.append({"shape": shape[i],
                                           "center": np.array(center_list[i]),
                                           "radius": np.array(radius_list[i])})
            elif shape[i] == "polygon":
                self.obstacle_list.append({"shape": shape[i],
                                           "vertices": np.array(vertices_list[i])})
            else:
                raise ValueError("Obstacle shape not given or not fit!")

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

    def cbf_calculate_obstacle_expansion(self, robot_position):
        expanded_shapes = []
        for i in range(len(self.obstacle_list)):
            if self.obstacle_list[i]["shape"] == "polygon":
                pass
            elif self.obstacle_list[i]["shape"] == "circle":
                expanded_shapes.append(casadi_distance(self.obstacle_list[i]["center"], robot_position))
            else:
                raise ValueError("Obstacle shape not given or not fit!")

        return expanded_shapes

    @staticmethod
    def cbf_expand_rate_2d(robot_position, segment):
        return ((robot_position[0]*(segment[1, 1]-segment[0, 1])-robot_position[1]*(segment[1, 0]-segment[0, 0]))
                / (segment[0, 0]*segment[1, 1]-segment[0, 1]*segment[1, 0]))


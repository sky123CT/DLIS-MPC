#from casadi import *
import numpy as np
from functional.CasadiDLIS import CasadiDLIS
from functional.cost_function import CSCostFunction
from functional.obstacle import Obstacle
from functional.car import Car
from functional.dynamics import Dynamics
from shapely.geometry import Point, Polygon

circle = Point(0, 0).buffer(10)
polygon = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
intersection = circle.intersection(polygon)
print(intersection)




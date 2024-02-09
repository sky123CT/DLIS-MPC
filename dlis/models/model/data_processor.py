import numpy as np
import openpyxl
from .utility import Utility


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def data_processing(data_path, obstacle_shape='circle'):
        wb = openpyxl.load_workbook(data_path)
        ws_is = wb['Initial States']
        ws_ob = wb['Obstacles']
        ws_ia = wb['Intersection Areas']

        data_is = Utility.xl_2_numpy(ws_is)
        data_ob = Utility.xl_2_numpy(ws_ob)
        data_ia = Utility.xl_2_numpy(ws_ia)
        if obstacle_shape == 'circle':
            rel_pos = Utility.relative_position_circle(data_is, data_ob)
            """
            inputs = np.concatenate((rel_pos,
                                     (np.arctan2(rel_pos[:, 1], rel_pos[:, 0])-data_is[:, 2]).reshape(-1, 1),
                                     data_ob[:, -1].reshape(-1, 1),
                                     (data_is[:, 3]+np.pi-data_is[:, 2]).reshape(-1, 1)
                                     ), axis=1)
            """
            inputs = np.concatenate((rel_pos,
                                     (data_ob[:, -1]).reshape(-1, 1),
                                     (data_is[:, 3]-data_is[:, 2]).reshape(-1, 1)
                                     ), axis=1)
        elif obstacle_shape == 'polygon':
            rel_pos = Utility.relative_position_polygon(data_is, data_ob)
            inputs = np.concatenate((rel_pos, data_is[:, 2:].reshape(data_is.shape[0], 2)), axis=1)
        else:
            raise ValueError('obstacle shape is not given!')
        labels = data_ia * 10000

        return inputs, labels

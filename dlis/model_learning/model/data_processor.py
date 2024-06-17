import numpy as np
import openpyxl
from model_learning.model.utility import Utility


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
            """
            inputs = np.concatenate((rel_pos,
                                     (data_ob[:, -1]**2).reshape(-1, 1),
                                     data_is[:, 2].reshape(-1, 1),
                                     data_is[:, 3].reshape(-1, 1)
                                     ), axis=1)
            """

            inputs = np.concatenate((rel_pos,
                                     (data_ob[:, -1] ** 2).reshape(-1, 1),
                                     -(data_is[:, 2].reshape(-1, 1)-data_is[:, 3].reshape(-1, 1))
                                     ), axis=1)

        elif obstacle_shape == 'polygon':
            rel_pos = Utility.relative_position_polygon(data_is, data_ob)
            inputs = np.concatenate((rel_pos, data_is[:, 2:].reshape(data_is.shape[0], 2)), axis=1)
        else:
            raise ValueError('obstacle shape is not given!')
        labels = data_ia * 10000

        return inputs, labels

    @staticmethod
    def data_processing_2trailer(data_path):
        wb = openpyxl.load_workbook(data_path)
        ws_initial_states = wb['Initial_States']
        ws_obstacles = wb['Obstacles']
        ws_intersections = wb['Intersection_Areas']

        data_initial_states = Utility.xl_2_numpy(ws_initial_states)
        data_obstacles = Utility.xl_2_numpy(ws_obstacles)
        data_intersections = Utility.xl_2_numpy(ws_intersections)

        relative_position1 = Utility.relative_position_circle(data_initial_states[:, :2], data_obstacles[:, :2])
        relative_position2 = Utility.relative_position_circle(data_initial_states[:, 3:5], data_obstacles[:, :2])

        """
        inputs = np.concatenate((rel_pos,
                                     (np.arctan2(rel_pos[:, 1], rel_pos[:, 0])-data_is[:, 2]).reshape(-1, 1),
                                     data_ob[:, -1].reshape(-1, 1),
                                     (data_is[:, 3]+np.pi-data_is[:, 2]).reshape(-1, 1)
                                     ), axis=1)    
        """
        """
        inputs = np.concatenate((rel_pos,
                                     (data_ob[:, -1]**2).reshape(-1, 1),
                                     data_is[:, 2].reshape(-1, 1),
                                     data_is[:, 3].reshape(-1, 1)
                                     ), axis=1)
        """

        inputs = np.concatenate((relative_position1,
                                 relative_position2,
                                 (data_obstacles[:, -2] ** 2).reshape(-1, 1),
                                 (data_obstacles[:, -1] ** 2).reshape(-1, 1),
                                 data_initial_states[:, 2].reshape(-1, 1),
                                 data_initial_states[:, 5].reshape(-1, 1),
                                 data_initial_states[:, 8].reshape(-1, 1)
                                 ), axis=1)
        labels = data_intersections * 10000

        return inputs, labels


if __name__ == '__main__':
    data_processor = DataProcessor()
    inputs, labels = data_processor.data_processing_2trailer(
        data_path='../data/dataset/sample_2trailers_10samples_with_tractor_1000points.xlsx')
    print(inputs)
    print(labels)

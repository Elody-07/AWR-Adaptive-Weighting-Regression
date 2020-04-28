import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import cv2

class VisualUtil:

    def __init__(self, dataset):
        self.dataset = dataset
        # RED BGR
        self.color_pred = [(0,0,102), (0,0,179), (0,0,255), (77,77,255), (153,153,255)]
        # self.color_pred = ['#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
        # BLUE BGR
        self.color_gt = [(102,0,0), (179,0,0), (255,0,0), (255,77,77), (255,153,153)]
        # self.color_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']

    def plot(self, img, path, jt_uvd_pred, jt_uvd_gt=None):
        uvd_pred = jt_uvd_pred.reshape(-1, 3)
        image = img.copy()
        image = (image.squeeze() + 1) * 100
        image = image[:, :, np.newaxis].repeat(3, axis=-1)

        self._plot_fingers(image, uvd_pred, self.color_pred)

        if isinstance(jt_uvd_gt, np.ndarray):
            uvd_gt = jt_uvd_gt.reshape(-1, 3)
            self._plot_fingers(image, uvd_gt, self.color_gt)
        cv2.imwrite(path, image)

    def _plot_fingers(self, img, jt_uvd, colors):
        jt_idx, sketch = self._get_setting()
        for i in range(len(colors)):
            for idx in jt_idx[i]:
                cv2.circle(img, (int(jt_uvd[idx][0]), int(jt_uvd[idx][1])),
                           2, colors[i], -1)

            for (s, e) in sketch[i]:
                cv2.line(img, (int(jt_uvd[s][0]), int(jt_uvd[s][1])),
                         (int(jt_uvd[e][0]), int(jt_uvd[e][1])),
                         colors[i], 1)
        return

    def _get_setting(self):
        if self.dataset == 'nyu':
            jt_idx = [[0,1], [2,3], [4,5], [6,7], [8,9,10,11,12,13]]
            sketch = [[(0, 1), (1, 13)],
                      [(2, 3), (3, 13)],
                      [(4, 5), (5, 13)],
                      [(6, 7), (7, 13)],
                      [(8, 9), (9, 10),(10, 13), (11, 13), (12, 13)]]
            return jt_idx, sketch

        elif 'hands' in self.dataset:
            jt_idx = [[1,6,7,8], [2,9,10,11], [3,12,13,14], [4,15,16,17], [5,18,19,20,0]]
            sketch = [[(0, 1), (1, 6), (6, 7), (7, 8)],
                      [(0, 2), (2, 9), (9, 10), (10, 11)],
                      [(0, 3), (3, 12), (12, 13), (13, 14)],
                      [(0, 4), (4, 15), (15, 16), (16, 17)],
                      [(0, 5), (5, 18), (18, 19), (19, 20)]]
            return jt_idx, sketch

        elif self.dataset == 'icvl':
            jt_idx = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15, 0]]
            sketch = [[(0, 1), (1, 2), (2, 3)],
                      [(0, 4), (4, 5), (5, 6)],
                      [(0, 7), (7, 8), (8, 9)],
                      [(0, 10), (10, 11), (11, 12)],
                      [(0, 13), (13, 14), (14, 15)]]
            return jt_idx, sketch

        elif self.dataset == 'msra':
            jt_idx = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20,0]]
            sketch = [[(0, 1), (1, 2), (2, 3), (3, 4)],
                      [(0, 5), (5, 6), (6, 7), (7, 8)],
                      [(0, 9), (9, 10), (10, 11), (11, 12)],
                      [(0, 13), (13, 14), (14, 15), (15, 16)],
                      [(0, 17), (17, 18), (18, 19), (19, 20)]]
            return jt_idx, sketch





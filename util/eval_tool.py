import numpy as np
import matplotlib.pyplot as plt
import os
from util.util import uvd2xyz

class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, img_size, paras, flip, num_kp):
        # init empty data storage
        self.data = list()
        self.img_size = img_size
        self.paras = paras
        self.flip = flip
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, jt_uvd_pred, jt_xyz_gt, center_xyz, M, cube, jt_vis=0, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            jt_uvd_pred = np.squeeze(jt_uvd_pred).astype(np.float32)
            jt_xyz_gt = np.squeeze(jt_xyz_gt).astype(np.float32)
            jt_vis = np.squeeze(jt_vis).astype('bool')
            center_xyz = np.squeeze(center_xyz).astype(np.float32)
            M = np.squeeze(M).astype(np.float32)
            cube = np.squeeze(cube).astype(np.float32)

            assert len(jt_uvd_pred.shape) == 2
            assert len(jt_xyz_gt.shape) == 2

        try:
            M_inv = np.linalg.inv(M)
        except:
            print('Inverse matrix does not exist.')

        jt_uvd_pred[:, :2] = (jt_uvd_pred[:, :2] + 1) * self.img_size / 2.
        jt_uvd_pred[:, 2] = jt_uvd_pred[:, 2] * cube[2] / 2. + center_xyz[2]
        jt_uvd_trans = np.hstack([jt_uvd_pred[:, :2], np.ones((jt_uvd_pred.shape[0], 1))])
        jt_uvd_pred[:, :2] = np.dot(M_inv, jt_uvd_trans.T).T[:, :2]
        jt_xyz_pred = uvd2xyz(jt_uvd_pred, self.paras, self.flip)

        jt_xyz_gt = jt_xyz_gt * (cube / 2.) + center_xyz

        # calc euclidean distance
        diff = jt_xyz_gt - jt_xyz_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = jt_xyz_gt.shape[0]
        for i in range(num_kp):
            if jt_vis == 0:
                self.data[i].append(euclidean_dist[i])
            else:
                if jt_vis[i]:
                    self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(0, 50, 100)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        # average mean/median error over num frames and joints
        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        # area under pck curve
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints
        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds

    def plot_pck(self, path, epoch, pck_curve_all, thresholds):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(thresholds, pck_curve_all * 100, '-*', label='model')
        ax.set_xlabel('threshold in mm')
        ax.set_ylabel('% of correct keypoints')
        plt.ylim([0.0, 100.0])
        plt.grid()
        plt.legend(loc='lower right')
        # plt.tight_layout(rect=(0.01, -0.05, 1.03, 1.03))
        plt.savefig(os.path.join(path, 'PCK_curve_epoch' + str(epoch) + '.png'))
        plt.close()


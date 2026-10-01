"""Self-contained ByteTrack association (adapted from official ByteTrack /
HybridSORT implementation; scipy matching replaces lap/cython_bbox).

Only reads video_id/frame_order/bbox/score/image size from the frozen
detection stream. No GT, no embeddings, no category labels.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """Standard 8-dim constant-velocity Kalman filter (from SORT/ByteTrack)."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[3], self._std_weight_position * mean[3],
                   1e-2, self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[3],
                   1e-5, self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3], self._std_weight_position * mean[3],
               1e-1, self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


_track_count = 0


def _next_id():
    global _track_count
    _track_count += 1
    return _track_count


class STrack:
    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.New
        self.track_id = -1
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        return self.tlwh_to_tlbr(self.tlwh)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = _next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.end_frame = frame_id
        if frame_id == 1:
            self.is_activated = True

    def re_activate(self, new_track, frame_id):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.end_frame = frame_id
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.end_frame = frame_id

    def mark_lost(self):
        self.state = TrackState.Lost
        self.end_frame = self.frame_id

    def mark_removed(self):
        self.state = TrackState.Removed


def iou_matrix(atlbrs, btlbrs):
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)))
    a = np.asarray(atlbrs, dtype=np.float64)
    b = np.asarray(btlbrs, dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def linear_assignment(cost, thresh):
    if cost.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost.shape[0])), tuple(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(rows, cols):
        if cost[r, c] <= thresh:
            matches.append([r, c])
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    return (np.asarray(matches, dtype=int),
            tuple(i for i in range(cost.shape[0]) if i not in matched_r),
            tuple(j for j in range(cost.shape[1]) if j not in matched_c))


def joint_stracks(tlista, tlistb):
    """Return the ordered union used by the reference ByteTrack update."""
    exists = {track.track_id for track in tlista}
    result = list(tlista)
    for track in tlistb:
        if track.track_id not in exists:
            exists.add(track.track_id)
            result.append(track)
    return result


def sub_stracks(tlista, tlistb):
    """Subtract tracks by identity, preserving input order."""
    remove = {track.track_id for track in tlistb}
    return [track for track in tlista if track.track_id not in remove]


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove near-identical tracks retained by the tracked/lost unions."""
    if not stracksa or not stracksb:
        return stracksa, stracksb
    distances = 1 - iou_matrix(
        [track.tlbr for track in stracksa],
        [track.tlbr for track in stracksb],
    )
    pairs = np.where(distances < 0.15)
    duplicates_a, duplicates_b = [], []
    for index_a, index_b in zip(*pairs):
        track_a, track_b = stracksa[index_a], stracksb[index_b]
        duration_a = track_a.frame_id - track_a.start_frame
        duration_b = track_b.frame_id - track_b.start_frame
        if duration_a > duration_b:
            duplicates_b.append(index_b)
        else:
            duplicates_a.append(index_a)
    return (
        [track for index, track in enumerate(stracksa) if index not in duplicates_a],
        [track for index, track in enumerate(stracksb) if index not in duplicates_b],
    )


def fuse_score(cost, detections):
    """Fuse IoU affinity with detection confidence as in ByteTrack."""
    if cost.size == 0 or not detections:
        return cost
    iou_similarity = 1 - cost
    scores = np.asarray([detection.score for detection in detections], dtype=np.float64)
    return 1 - iou_similarity * scores[None, :]


class BYTETracker:
    """ByteTrack association on frozen detection frames."""

    def __init__(self, track_thresh=0.5, low_thresh=0.1, match_thresh=0.8,
                 track_buffer=30, frame_rate=30, min_box_area=10,
                 mot20=False):
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.track_buffer
        self.min_box_area = min_box_area
        self.mot20 = mot20
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.kalman_filter = KalmanFilter()

    def reset(self):
        global _track_count
        _track_count = 0
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

    def update(self, dets_xyxy_score):
        self.frame_id += 1
        output_results = np.asarray(dets_xyxy_score, dtype=np.float64)
        if output_results.shape[0] == 0:
            output_results = np.empty((0, 5))
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        remain = scores > self.track_thresh
        inds_low = scores > self.low_thresh
        inds_high = scores <= self.track_thresh
        inds_second = inds_low & inds_high
        dets = bboxes[remain]
        scores_keep = scores[remain]
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        detections = [STrack(self.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets, scores_keep)]
        detections_second = [STrack(self.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_second, scores_second)]

        unconfirmed = []
        tracked = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                unconfirmed.append(t)
            else:
                tracked.append(t)
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        for t in strack_pool:
            t.predict()

        dists = 1 - iou_matrix([t.tlbr for t in strack_pool], [t.tlbr for t in detections])
        if not self.mot20:
            dists = fuse_score(dists, detections)
        matches, u_track, u_det = linear_assignment(dists, self.match_thresh)
        activated, refind = [], []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists2 = 1 - iou_matrix([t.tlbr for t in r_tracked], [t.tlbr for t in detections_second])
        matches2, u_track2, u_det2 = linear_assignment(dists2, 0.5)
        for itracked, idet in matches2:
            track = r_tracked[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)
        lost_now = []
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_now.append(track)

        # Match one-frame/unconfirmed tracks against unmatched high-score
        # detections before creating new tracks (ByteTrack's third stage).
        unmatched_detections = [detections[index] for index in u_det]
        dists3 = 1 - iou_matrix(
            [track.tlbr for track in unconfirmed],
            [detection.tlbr for detection in unmatched_detections],
        )
        if not self.mot20:
            dists3 = fuse_score(dists3, unmatched_detections)
        matches3, u_unconfirmed, u_detection = linear_assignment(dists3, 0.7)
        for itracked, idet in matches3:
            track = unconfirmed[itracked]
            track.update(unmatched_detections[idet], self.frame_id)
            activated.append(track)
        unmatched_unconfirmed = set(u_unconfirmed)
        removed_now = []
        for index, track in enumerate(unconfirmed):
            if index in unmatched_unconfirmed:
                track.mark_removed()
                removed_now.append(track)

        # Only detections above the reference new-track threshold start a new
        # identity. Lower-confidence detections are association evidence only.
        for index in u_detection:
            track = unmatched_detections[index]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        # Expire lost tracks by frame age, not by the number of lost tracks.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_now.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks.extend(lost_now)
        self.removed_stracks.extend(removed_now)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # output: bboxes that were tracked this frame
        out_map = {}
        for t in self.tracked_stracks:
            if t.frame_id == self.frame_id and t.is_activated:
                tlbr = t.tlbr
                tlwh = t.tlwh
                vertical = tlwh[2] / max(tlwh[3], 1e-6) > 1.6
                if self.mot20 or (tlwh[2] * tlwh[3] > self.min_box_area and not vertical):
                    out_map[t.track_id] = [tlbr[0], tlbr[1], tlbr[2], tlbr[3], t.track_id, t.score]
        out = list(out_map.values())
        return np.asarray(out, dtype=np.float64).reshape(-1, 6)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        return STrack.tlbr_to_tlwh(tlbr)

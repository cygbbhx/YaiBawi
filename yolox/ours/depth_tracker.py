import numpy as np
from yolox.tracker import matching
from yolox.tracker.basetrack import BaseTrack, TrackState
from yolox.tracker.byte_tracker import STrack, BYTETracker, joint_stracks, sub_stracks, remove_duplicate_stracks
from yolox.deepsort_tracker import kalman_filter, linear_assignment, iou_matching
from yolox.deepsort_tracker.reid_model import Extractor
from yolox.tracker import matching
import networkx as nx

def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


class OurDetection(STrack):
    def __init__(self, tlwh, score):
        super().__init__(tlwh, score)
        self.feature = None
        self.occluded = False
        self.previous_depth = self._tlwh[1] + self._tlwh[-1]
        self.neighbors = set()
    
    def set_feature(self, feat):
        self.feature = feat

    def update_direct(self, kalman_filter, new_track, frame_id):
        self.update(new_track, frame_id)
        self.kalman_filter = kalman_filter
        new_tlwh = new_track._tlwh
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(new_tlwh))

    def prepare_rematch(self, neighbor):
        self.occluded = True
        self.previous_depth = self.tlwh[1] + self.tlwh[-1]
        self.neighbors.add(neighbor)

    def set_not_occluded(self):
        self.occluded = False
        self.neighbors = set()

    def current_depth(self):
        return self.tlwh[1] + self.tlwh[-1]

class ReIdTracker(BYTETracker):
    def __init__(self, args, frame_rate=30, max_dist=0.1, nn_budget=100):
        super().__init__(args, frame_rate=frame_rate)
        self.extractor = Extractor('pretrained/ckpt.t7', use_cuda=True)
        max_cosine_distance = max_dist
        self.metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            # scores = output_results[:, 4] * output_results[:, 5]
            # bboxes = output_results[:, :4]  # x1y1x2y2
            scores = output_results.conf
            bboxes = output_results.xyxy # x1y1x2y2 
        img_h, img_w = img_info[0], img_info[1]
        raw_img = img_info[2]
        self.height, self.width = raw_img.shape[:2]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores >= self.args.track_thresh
        inds_low = scores > 0.6
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        reused_detections = False

        if len(dets) > 0:
            '''Detections'''
            detections = [OurDetection(OurDetection.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
            features = self._get_features(dets, raw_img)
            for i, det in enumerate(detections):
                det.set_feature(features[i])
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[OurDetection]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        OurDetection.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if self.frame_id > 5 and len(u_detection) > 0:
            reused_detections = True
            inds_second = [(i in u_detection) or value for i, value in enumerate(inds_second)]
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [OurDetection(OurDetection.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
            features_second = self._get_features(dets_second, raw_img)
            for i, det in enumerate(detections_second):
                det.set_feature(features_second[i])
        else:
            detections_second = []
            features_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        targets = [t.track_id for t in r_tracked_stracks]

        # if len(detections_second) > 0:
            # print(f'rematch in frame {self.frame_id}')
        # depth_dists = depth_distance(r_tracked_stracks, detections_second)
        dists = self.metric.distance(features_second, targets)
        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.args.second_thresh)
        # matches, u_track, u_detection_second = matching.linear_assignment(depth_dists, thresh=self.args.second_thresh)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                # track.update(det, self.frame_id)
                track.update_direct(self.kalman_filter, det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        final_u_detection = u_detection_second if reused_detections else u_detection
        final_detections = detections_second if reused_detections else detections
        detections = [final_detections[i] for i in final_u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, final_u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in final_u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # # Double check previously occluded tracks
        # revised_stracks = [track for track in self.tracked_stracks if track.occluded]

        # if len(revised_stracks) > 0:
        #     depth_dists = depth_distance(revised_stracks, revised_stracks, local=True)
        #     matches, _, _ = matching.linear_assignment(depth_dists, thresh=np.inf) # bipartite matching, no match cannot exist
            
        #     for itracked, idet in matches:
        #         if itracked == idet:
        #             continue
        #         self.tracked_stracks[itracked].update_direct(self.kalman_filter, revised_stracks[idet], self.frame_id)
        
        # # Upate Occlusion Status
        # tracked_bboxes = [track.tlwh for track in self.tracked_stracks]
        # iou_distances = calculate_iou_matrix(tracked_bboxes) 

        # occlusion_groups = find_occlusions(iou_distances, iou_threshold=self.args.iou_thresh)
        # occluded_inst = sum(occlusion_groups, [])
        # for idx in range(len(self.tracked_stracks)):
        #     if idx not in occluded_inst:
        #         self.tracked_stracks[idx].set_not_occluded()
        
        # for group in occlusion_groups:
        #     print(f'rematch in frame {self.frame_id}')
        #     print(f'rematching {group}')
        #     for i in range(len(group)):
        #         for j in range(i+1, len(group)):
        #             reid_pair = [self.tracked_stracks[i], self.tracked_stracks[j]]
        #             depth_dists = depth_distance(reid_pair, reid_pair)
        #             matches, _, _ = matching.linear_assignment(depth_dists, thresh=np.inf) # bipartite matching, no match cannot exist
                    
        #     for itracked, idet in matches:
        #         if itracked == idet:
        #             continue
        #         self.tracked_stracks[itracked].update_direct(self.kalman_filter, reid_pair[idet], self.frame_id)

        #     for i in range(len(group)):
        #         for j in range(i+1, len(group)):
        #             self.tracked_stracks[i].prepare_rematch(self.tracked_stracks[j].track_id)
        #             self.tracked_stracks[j].prepare_rematch(self.tracked_stracks[i].track_id)

        tracked_features = [t.feature for t in self.tracked_stracks]
        targets = [t.track_id for t in self.tracked_stracks]
        active_targets = [t.track_id for t in output_stracks]

        self.metric.partial_fit(
            np.asarray(tracked_features), np.asarray(targets), active_targets)

        return output_stracks

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

def depth_distance(depths_a, depths_b, local=False):
    """
    Compute cost based on depth differences.
    :type depths_a: list[float] | np.ndarray
    :type depths_b: list[float] | np.ndarray

    :rtype: np.ndarray
    """
    costs = np.zeros((len(depths_a), len(depths_b)), dtype=np.float)
    if costs.size == 0:
        return costs

    for i, t_a in enumerate(depths_a):
        for j, t_b in enumerate(depths_b):
            if local:
                if t_b.track_id not in t_a.neighbors and t_b.track_id != t_a.track_id:
                    costs[i, j] = 2000
                    continue
            # else:
                # costs[i, j] = abs(t_a.previous_depth - t_b.current_depth())
            costs[i, j] = abs(t_a.previous_depth - t_b.current_depth())

    return costs


def find_occlusions(cost_matrix, iou_threshold):
    num_detections = cost_matrix.shape[0]
    G = nx.Graph()

    for i in range(num_detections):
        G.add_node(i)

    for i in range(num_detections):
            for j in range(i + 1, num_detections):
                if cost_matrix[i, j] > 0.1:
                    G.add_edge(i, j, weight=cost_matrix[i,j])

    initial_occlude_groups = [list(component) for component in nx.connected_components(G)]

    occlude_groups = []
    for group in initial_occlude_groups:
        if len(group) < 2:
            continue
        
        subgraph = G.subgraph(group)
        total_edge_weight = sum(data['weight'] for u, v, data in subgraph.edges(data=True))
        
        if len(group) == 2 and total_edge_weight >= iou_threshold:
            occlude_groups.append(group)
        else:
            occlude_groups.append(group)

    return occlude_groups

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def calculate_iou_matrix(boxes):
    num_boxes = len(boxes)
    iou_matrix = np.zeros((num_boxes, num_boxes))

    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j:
                iou_matrix[i, j] = calculate_iou(boxes[i], boxes[j])

    return iou_matrix

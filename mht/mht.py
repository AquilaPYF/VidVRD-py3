import json
from collections import defaultdict

from scipy.spatial.distance import mahalanobis

from mht.tree import TrackTree, TreeNode

from mht.track_utils import check_2_nodes

import numpy as np

from mht.relation import VideoRelation

from mht.trajectory import Trajectory


def origin_mht_relational_association(short_term_relations, truncate_per_segment=100, top_tree=3):
    """
    This is not the very official MHT framework, which mainly is 4 frame-level.
    This func is to associating short-term-relations relational.
    :param top_tree:
    :param short_term_relations:
    :param truncate_per_segment:
    :return:
    """
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    tree_dict = dict()

    for pstart in sorted(pstart_relations.keys()):
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]

        # Traversing truncate_per_segment relations
        for each_rela in sorted_relations:
            subj, pred_rela, obj = each_rela['triplet']
            score = each_rela['score']
            subj_tracklet = each_rela['sub_traj']
            obj_tracklet = each_rela['obj_traj']
            duration = each_rela['duration']
            each_triplet = (subj, pred_rela, obj)  # This is tree name

            if each_triplet not in tree_dict.keys():
                tree_dict[each_triplet] = TrackTree()
                new_tree_node = TreeNode(triplet=each_triplet,
                                         score=score,
                                         subj_tracklet=subj_tracklet,
                                         obj_tracklet=obj_tracklet,
                                         duration=duration)
                tree_dict[each_triplet].add(new_tree_node)
            else:
                track_tree = tree_dict[each_triplet]
                new_tree_node = TreeNode(triplet=each_triplet,
                                         score=score,
                                         subj_tracklet=subj_tracklet,
                                         obj_tracklet=obj_tracklet,
                                         duration=duration)

                if duration[0] == 0:
                    track_tree.add(new_tree_node)
                else:
                    for each_path in track_tree.get_paths():
                        if check_2_nodes(each_path[-1], new_tree_node):
                            track_tree.add(new_tree_node, each_path[-1])
    # track Scoring

    # global hypothesis formation

    # track tree pruning

    # generate results
    video_relation_list = list()
    for each_triplet, each_tree in tree_dict.items():
        save_res_path = 'test_out.json'
        top_k_paths, top_k_scores = generate_results(each_tree, top_tree)
        for each_path in top_k_paths:
            video_relation_list.append(associate_path(each_path))
    return [r.serialize() for r in video_relation_list]


def get_gating(pre_traj, distance_threshold=0.5):
    """
    Where the next observation of the track is expected to appear. Use mahalanobis distance.
    :param pre_traj: track 2 b predicted
    :param distance_threshold: distance threshold
    :return: a gating area where the next observation of the track is expected 2 appear.
    """
    pre_location = np.array(pre_traj[-1])
    predict = np.array(gating_predict(pre_traj))
    distance = np.dot((predict - pre_location).T, )
    mahalanobis_distance = mahalanobis(pre_location, predict, pre_traj)
    if mahalanobis_distance <= distance_threshold:
        return True
    return False


def gating_predict(pre_traj):
    predict_result = pre_traj[-1]
    return predict_result


def track_score(track_path):
    """
    Scoring the proposal tracklet can b associate 2 st_track possibility.
    Score = weight_motion * score_motion + weight_appearance * score_appearance
    :return:
    """
    weight_motion, score_motion, weight_appearance, score_appearance = 0, 0, 0, 0
    score = weight_motion * score_motion + weight_appearance * score_appearance

    path_score = 0.
    for each_node in track_path:
        path_score += each_node.score
    return path_score / len(track_path)


def global_hypo(track_trees):
    """
    Determine the most likely combination of object tracks at frame k.
    NP-hard, Maximum Weighted Independent Set Problem (MWIS)
    :param track_trees: a set of trees containing all traj hypotheses 4 all targets
    :return:
    """


def generate_results(track_tree, top_k):
    """
    :param top_k:
    :param track_tree:
    :return:
    """
    path_score_dict = dict()
    for each_path in track_tree.get_paths():
        path_score_dict[track_score(each_path)] = each_path
    sorted_keys = sorted(path_score_dict.keys(), reverse=True)
    top_k_res = list()
    if len(sorted_keys) >= top_k:
        for i in range(top_k):
            top_k_res.append(path_score_dict[sorted_keys[i]])
        top_k_scores = sorted_keys[:top_k]
    else:
        for each_key in sorted_keys:
            top_k_res.append(path_score_dict[each_key])
        top_k_scores = sorted_keys
    return top_k_res, top_k_scores


def associate_path(track_path):
    result = None
    for each_node in track_path:
        if each_node.duration != [0, 0]:
            sub, pred, obj = each_node.triplet
            pstart, pend = each_node.duration
            conf_score = each_node.score
            straj = Trajectory(pstart, pend, each_node.subj_tracklet, conf_score)
            otraj = Trajectory(pstart, pend, each_node.obj_tracklet, conf_score)

            if result is None:
                result = VideoRelation(sub, pred, obj, straj, otraj, conf_score)
            else:
                result.extend(straj, otraj, conf_score)
    return result


if __name__ == '__main__':
    with open('test.json', 'r') as test_st_rela_f:
        test_st_rela = json.load(test_st_rela_f)

    result = origin_mht_relational_association(test_st_rela)

    print(len(result))
    for each_res in result:
        print(each_res)

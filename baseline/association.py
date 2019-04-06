import json
from collections import defaultdict

from baseline.trajectory import Trajectory
from baseline.relation import VideoRelation


def greedy_relational_association(short_term_relations, truncate_per_segment=100):
    # group short-term relations by their staring frames
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    video_relation_list = []
    last_modify_rel_list = []
    for pstart in sorted(pstart_relations.keys()):
        last_modify_rel_list.sort(key=lambda r: r.score(), reverse=True)
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]

        cur_modify_rel_list = []
        for rel in sorted_relations:
            conf_score = rel['score']
            sub, pred, obj = rel['triplet']
            _, pend = rel['duration']
            straj = Trajectory(pstart, pend, rel['sub_traj'])
            otraj = Trajectory(pstart, pend, rel['obj_traj'])

            for r in last_modify_rel_list:
                if r.triplet() == tuple(rel['triplet']) and r.both_overlap(straj, otraj, iou_thr=0.5):
                    # merge
                    r.extend(straj, otraj, conf_score)
                    last_modify_rel_list.remove(r)
                    cur_modify_rel_list.append(r)
                    break
            else:
                r = VideoRelation(sub, pred, obj, straj, otraj, conf_score)
                video_relation_list.append(r)
                cur_modify_rel_list.append(r)

        last_modify_rel_list = cur_modify_rel_list

    return [r.serialize() for r in video_relation_list]

from baseline.track_tree import TrackTree, TreeNode

def origin_mht_relational_association(short_term_relations, truncate_per_segment=100):
    """
    This is not the very official MHT framework, which mainly is 4 frame-level.
    This func is to associating short-term-relations relational.
    ECCV 2018 update this original MHT 2 RNNs-Gating Network.
    :param short_term_relations:
    :param truncate_per_segment:
    :return:
    """
    # Step 0. construct & update track tree
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    targets_trees = dict()
    targets_labels = set()
    for pstart in sorted(pstart_relations.keys()):
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]

        for each_rela in sorted_relations:
            subj, pred_rela, obj = each_rela['triplet']
            for target in [subj, obj]:
                if target in targets_labels:
                    # get all of same label trees
                    target_tree_list = []
                    current_num = 0
                    for each_node in targets_trees.keys():
                        label, id = each_node.split('#')
                        if target == label:
                            current_num = max(current_num, id)
                            target_tree_list.append(each_node)
                    # figure out whether update the tree or create a new one
                    update_nodes = []
                    # generate update nodes ...

                    if len(update_nodes) > 0:
                        # update trees
                        for each_update_node in update_nodes:
                            # update tracklet
                            pass
                    else:
                        # create a new tree
                        tree_name = target + '#' + str(current_num + 1)


                else:
                    # create a new tree
                    targets_labels.add(target)
                    tree_name = target + '#0'




    # Step 1. Gating

    # Step 2. Track Scoring
    # Step 3. Global Hypothesis Formation
    # Step 4. Track Tree Pruning


def gating(st_track, dth):
    """
    Where the next observation of the track is expected to appear.
    :param st_track: track 2 b predicted
    :param dth: distance threshold
    :return: a gating area where the next observation of the track is expected 2 appear.
    """


def track_score(st_track, proposal):
    """
    Scoring the proposal tracklet can b associate 2 st_track possibility
    :param st_track: exist track (traj)
    :param proposal: tracklet 2 b connected
    :return: score
    """


def global_hypo(track_trees):
    """

    :param track_trees: a set of trees containing all traj hypotheses 4 all targets
    :return:
    """


def pruning_track_tree(track_tree):
    """
    Track Tree Pruning
    :param track_tree:
    :return:
    """


def generate_results(track_trees):
    """
    Generate traj results finally.
    :param track_trees:
    :return: trajs
    """


if __name__ == '__main__':
    rpath = '/home/daivd/PycharmProjects/VidVRD-py3/'

    short_term_relations_path = rpath + 'baseline/vidvrd-dataset/vidvrd-baseline-output/short-term-predication.json'

    # with open(short_term_relations_path, 'r') as st_rela_in:
    #     short_term_relations = json.load(st_rela_in)

    # result = greedy_relational_association(short_term_relations['ILSVRC2015_train_00010001'])

    with open('test.json', 'r') as test_st_rela_f:
        test_st_rela = json.load(test_st_rela_f)

    result = origin_mht_relational_association(test_st_rela)

    # print(result)

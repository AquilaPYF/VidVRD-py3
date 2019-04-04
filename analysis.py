import json
import os
from collections import defaultdict

import numpy as np

from dataset import VidVRD
from evaluation.common import voc_ap
from evaluation.visual_relation_detection import eval_tagging_scores, eval_detection_scores
import matplotlib.pyplot as plt


def analyz_4_seg(seg, s_tra=False, o_tra=False, pred=False, s_label=False, o_label=False):
    """
    This func is try to evaluate performance on each segment, and calculate mean performance overall.
    :return:
    """
    # Analyz 4 traklets
    seg = analyz_4_traklet(seg, s_tra, o_tra)
    # Analyz 4 predicate
    seg = analyz_4_predicate(seg, pred)
    # Analyz 4 solabel
    seg = analyze_4_solabel(seg, s_label, o_label)

    return seg


def analyz_4_traklet(seg, s_tra=False, o_tra=False):
    """
    If using the gt tracklets as input, how much will improve.
    :return:
    """
    seg_result = dict()
    if not s_tra and not o_tra:
        seg_result = seg
    if s_tra:
        # Correct subject tracklet
        pass
    if o_tra:
        # Correct object tracklet
        pass

    return seg_result


def analyz_4_predicate(seg, pred=False):
    """
    If all of predicate are correct, how much will improve.
    :return:
    """
    seg_result = dict()
    if pred:
        # Correct predicate
        pass
    return seg_result


def analyze_4_solabel(seg, gt, s_label=False, o_label=False):
    """
    If all of so labels are correct.
    :return:
    """
    seg_result = dict()
    if not s_label and not o_label:
        seg_result = seg

    gt_s, gt_rela, gt_o = gt['triplet']

    if s_label:
        # Correct subject label
        pass
    if o_label:
        # Correct object label
        pass

    return seg_result


def analyze_4_associate():
    """
    Prove improving associate is important.
    :return:
    """
    return


def evaluate_segs(groundtruth, prediction, relation, viou_threshold=0.5,
                  det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """
    evaluate VRD on segments level
    :return:
    """
    print('Computing average precision AP over {}({}) videos...'.format('prediction', len(prediction)))
    print('This evaluation is based segments, traversal on predictions...')

    seg_ap = list()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0
    for vid, gt_relations in groundtruth.items():
        for each_gt_relations in separate_vid_2_seg(gt_relations):
            for each_gt_rela in each_gt_relations:
                if relation not in each_gt_rela['triplet'][1]:
                    each_gt_relations.remove(each_gt_rela)
            if len(each_gt_relations) == 0:
                continue
            tot_gt_relations += len(each_gt_relations)
            seg_duration = each_gt_relations[0]['duration']
            predict_relations = []
            for each_pred_rela in prediction[vid]:
                if each_pred_rela['duration'] == seg_duration and relation in each_pred_rela['triplet'][1]:
                    predict_relations.append(each_pred_rela)

            # compute average precision and recalls in detection setting
            det_prec, det_rec, det_scores = eval_detection_scores(
                each_gt_relations, predict_relations, viou_threshold)
            seg_ap.append(voc_ap(det_rec, det_prec))
            tp = np.isfinite(det_scores)
            for nre in det_nreturns:
                cut_off = min(nre, det_scores.size)
                tot_scores[nre].append(det_scores[:cut_off])
                tot_tp[nre].append(tp[:cut_off])
            # compute precisions in tagging setting
            tag_prec, _, _ = eval_tagging_scores(each_gt_relations, predict_relations)
            for nre in tag_nreturns:
                cut_off = min(nre, tag_prec.size)
                if cut_off > 0:
                    prec_at_n[nre].append(tag_prec[cut_off - 1])
                else:
                    prec_at_n[nre].append(0.)

    # calculate mean ap for detection
    mean_ap = np.mean(seg_ap)
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        if len(rec) >= 1:
            rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])

    # print scores
    print('detection mean AP (used in challenge): {}'.format(mean_ap))
    # if len(rec_at_n) > 50:
    #     print('detection recall@50: {}'.format(rec_at_n[50]))
    # if len(rec_at_n) > 100:
    #     print('detection recall@100: {}'.format(rec_at_n[100]))
    # print('tagging precision@1: {}'.format(mprec_at_n[1]))
    # if len(mprec_at_n) > 5:
    #     print('tagging precision@5: {}'.format(mprec_at_n[5]))
    # if len(mprec_at_n) > 10:
    #     print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mean_ap, rec_at_n, mprec_at_n


def separate_vid_2_seg(gt_relations):
    """
    :param gt_relations:
    :return:
    """
    # find out max seg end
    max_seg_end = 0
    for each_rela_ins in gt_relations:
        each_ins_seg_end = each_rela_ins['duration'][1]
        if max_seg_end < each_ins_seg_end:
            max_seg_end = each_ins_seg_end

    seg_list = []
    for start_f in range(0, max_seg_end - 15, 15):
        end_f = start_f + 30
        seg_insts = []

        for each_rela_ins in gt_relations:
            each_ins_s, each_ins_e = each_rela_ins['duration']
            if each_ins_s <= start_f and end_f <= each_ins_e:
                seg_ins_s_traj = each_rela_ins['sub_traj'][start_f - each_ins_s: end_f - each_ins_s]
                seg_ins_o_traj = each_rela_ins['obj_traj'][start_f - each_ins_s: end_f - each_ins_s]
                seg_ins = {
                    "triplet": each_rela_ins['triplet'],
                    "subject_tid": each_rela_ins['subject_tid'],
                    "object_tid": each_rela_ins['object_tid'],
                    "duration": [start_f, end_f],
                    "sub_traj": seg_ins_s_traj,
                    "obj_traj": seg_ins_o_traj
                }
                seg_insts.append(seg_ins)
        seg_list.append(seg_insts)
    return seg_list


def evaluate_relation(dataset, split, prediction, relation, segment=False):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)

    if segment:
        mean_ap, rec_at_n, mprec_at_n = evaluate_segs(groundtruth, prediction, relation)

    # # evaluate in zero-shot setting, if u need
    # print('-- zero-shot setting')
    # zeroshot_triplets = dataset.get_triplets(split).difference(
    #     dataset.get_triplets('train'))
    # groundtruth = dict()
    # zs_prediction = dict()
    # for vid in dataset.get_index(split):
    #     gt_relations = dataset.get_relation_insts(vid)
    #     zs_gt_relations = []
    #     for r in gt_relations:
    #         if tuple(r['triplet']) in zeroshot_triplets:
    #             zs_gt_relations.append(r)
    #     if len(zs_gt_relations) > 0:
    #         groundtruth[vid] = zs_gt_relations
    #         zs_prediction[vid] = []
    #         for r in prediction[vid]:
    #             if tuple(r['triplet']) in zeroshot_triplets:
    #                 zs_prediction[vid].append(r)
    # if segment:
    #     mean_ap, rec_at_n, mprec_at_n = evaluate_segs(groundtruth, zs_prediction, relation)
    return mean_ap, rec_at_n, mprec_at_n


def visualization():
    anno_rpath = 'baseline/vidvrd-dataset'
    with open(os.path.join(anno_rpath, 'test_relations.json'), 'r') as rela_f:
        relations = json.load(rela_f)
    with open('seg_res.json', 'r') as res_f:
        seg_res = json.load(res_f)

    fontsize = 30
    plt.figure(figsize=(50, 12))

    first_rela = relations['first']
    second_rela = relations['second']
    third_rela = relations['third']

    color_list = []
    y_sum_list = []
    x_rela_list = []
    seg_res = sorted(seg_res.items(), key=lambda item: item[1], reverse=True)
    for each_seg_rela in seg_res:
        if each_seg_rela[0] in first_rela:
            color_list.append('#FFA07A')
        if each_seg_rela[0] in second_rela:
            color_list.append('#87CEFA')
        if each_seg_rela[0] in third_rela:
            color_list.append('#FFBFDF')

        y_sum_list.append(each_seg_rela[1])
        x_rela_list.append(each_seg_rela[0])

    plt.bar(range(len(x_rela_list)), y_sum_list, color=color_list, tick_label=x_rela_list)
    plt.bar(0, 0, color='#FAA460', label='Actions')
    plt.bar(0, 0, color='#87CEFA', label='Spatio-relations')
    plt.bar(0, 0, color='#FFBFDF', label='Special-prep')
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.axis('tight')
    plt.xlim([-1, len(x_rela_list)])
    plt.tight_layout()
    plt.legend(loc="upper right", prop={'size': 30})
    plt.savefig("{}.png".format('test split'), dpi=200)
    plt.show()


if __name__ == '__main__':
    # anno_rpath = 'baseline/vidvrd-dataset'
    # video_rpath = 'baseline/vidvrd-dataset/videos'
    # splits = ['test']
    # st_prediction = 'baseline/vidvrd-dataset/vidvrd-baseline-output/short-term-predication.json'
    #
    # dataset = VidVRD(anno_rpath=anno_rpath,
    #                  video_rpath=video_rpath,
    #                  splits=splits)
    #
    # print('Loading prediction from {}'.format(st_prediction))
    # with open(st_prediction, 'r') as fin:
    #     pred = json.load(fin)
    #
    # with open(os.path.join(anno_rpath, 'test_relations.json'), 'r') as rela_f:
    #     relations = json.load(rela_f)
    #
    # res = dict()
    # for each_pos in ['first', 'second', 'third']:
    #     for rela in relations[each_pos]:
    #         print('Now is evaluating:', rela)
    #
    #         mAP, _, _ = evaluate_relation(dataset, 'test', pred, rela, segment=True)
    #         res[rela] = mAP
    #
    # print(res)
    #
    # with open('seg_res.json', 'w+') as out_f:
    #     out_f.write(json.dumps(res))

    visualization()

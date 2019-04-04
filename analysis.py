import json
import os
from evaluate import eval_visual_relation, eval_visual_relation_segs

dataset_rpath = 'baseline/vidvrd-dataset'
groundtruth_rpath = os.path.join(dataset_rpath, 'test')

# This is short-term (segment) predicate results
st_predict_results_path = os.path.join(dataset_rpath, 'vidvrd-baseline-output/short-term-predication.json')

# This is the result of association.
associate_pred_results_path = os.path.join(dataset_rpath,
                                           'vidvrd-baseline-output/models/baseline_relation_predication.json')


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


def analyze_4_solabel(seg, s_label=False, o_label=False):
    """
    If all of so labels are correct.
    :return:
    """
    seg_result = dict()
    if not s_label and not o_label:
        seg_result = seg
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


if __name__ == '__main__':
    with open(st_predict_results_path, 'r') as pred_in_f:
        print('Loading short-term prediction results ......')
        st_predict_results = json.load(pred_in_f)

    # Nothing Changed, evaluate performance of segment
    for each_vid in st_predict_results.keys():
        print('Now is evaluating: \t ' + each_vid)
        for each_seg in st_predict_results[each_vid]:
            pred_seg = analyz_4_seg(each_seg)
            eval_visual_relation_segs()

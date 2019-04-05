import json
from dataset import VidVRD
from evaluation.visual_relation_detection import separate_vid_2_seg
from baseline import segment_video
from baseline import trajectory, feature, model, association
from baseline import segment_video, get_model_path, get_segment_signature
from evaluation import eval_visual_relation
from dataset import VidVRD

# with open('gt_segs1.json', 'r') as in_f:
#     gt_segs = json.load(in_f)
#
# with open('gt_segs2.json', 'r') as in_f:
#     gt_segs2 = json.load(in_f)
#
#
# print(len(gt_segs.keys()))
# print(len(gt_segs2.keys()))
# gt_segs_keys = set(gt_segs.keys())
# gt_segs2_keys = set(gt_segs2.keys())
#
# print(len(gt_segs_keys), len(gt_segs2_keys))
#
# for id, segs in gt_segs2.items():
#     print(id)
#     print(len(segs))
#     print(gt_segs[id])
#     break


anno_rpath = 'baseline/vidvrd-dataset'
video_rpath = 'baseline/vidvrd-dataset/videos'
splits = ['train', 'test']
prediction = 'baseline/vidvrd-dataset/vidvrd-baseline-output/models/baseline_relation_prediction.json'
st_prediction = 'baseline/vidvrd-dataset/vidvrd-baseline-output/short-term-predication.json'

dataset = VidVRD(anno_rpath=anno_rpath,
                 video_rpath=video_rpath,
                 splits=splits)

video_indices = dataset.get_index(split='test')

with open(st_prediction, 'r') as st_pre_f:
    pred_segs = json.load(st_pre_f)

short_term_gt = dict()
short_term_pred = dict()

for vid in video_indices:
    gt = dataset.get_relation_insts(vid)
    pred = pred_segs[vid]
    gt_segs = separate_vid_2_seg(gt)

    for each_gt_seg in gt_segs:
        if len(each_gt_seg) < 1:
            continue
        fstart, fend = each_gt_seg[0]['duration']
        vsig = get_segment_signature(vid, fstart, fend)
        short_term_gt[vsig] = each_gt_seg

        for each_pred_seg in pred:
            if each_pred_seg['duration'] == each_gt_seg[0]['duration']:
                if vsig in short_term_pred.keys():
                    short_term_pred[vsig].append(each_pred_seg)
                else:
                    short_term_pred[vsig] = [each_pred_seg]

with open('gt_segs.json', 'w+') as out_f:
    out_f.write(json.dumps(short_term_gt))

with open('pred_segs.json', 'w+') as out_f:
    out_f.write(json.dumps(short_term_pred))

with open('gt_segs.json', 'r') as in_f:
    short_term_gt = json.load(in_f)

with open('pred_segs.json', 'r') as in_f:
    short_term_pred = json.load(in_f)

for each_vsig in short_term_gt.keys():
    if each_vsig not in short_term_pred.keys():
        short_term_pred[each_vsig] = []

mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(short_term_gt, short_term_pred)

print('detection mean AP (used in challenge): {}'.format(mean_ap))
print('detection recall@50: {}'.format(rec_at_n[50]))
print('detection recall@100: {}'.format(rec_at_n[100]))
print('tagging precision@1: {}'.format(mprec_at_n[1]))
print('tagging precision@5: {}'.format(mprec_at_n[5]))
print('tagging precision@10: {}'.format(mprec_at_n[10]))

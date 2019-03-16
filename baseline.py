import json
import os

from tqdm import tqdm

from baseline import segment_video, get_model_path
from baseline import trajectory, feature, model, association
from dataset import VidVRD

# could modify these paths
anno_rpath = 'baseline/vidvrd-dataset'
video_rpath = os.path.join(anno_rpath, 'videos')
splits = ['train', 'test']

# If u need 2 test u own short-term-predication, set this True
need_2_re4mat = True

# Put u short-term-predication json file 2
#  'baseline/vidvrd-dataset/vidvrd-baseline-output/result.json'
raw_result_json_file = 'result.json'    # U own json file name, could modify


if need_2_re4mat:
    stp_results_root_path = os.path.join(anno_rpath, 'vidvrd-baseline-output')
    with open(os.path.join(stp_results_root_path, raw_result_json_file), 'r') as in_f:
        with open(os.path.join(stp_results_root_path, 'results_re4mat.json'), 'w+') as out_f:
            out_f.write(json.dumps(json.load(in_f)['results']))

# short_term_predication_path = os.path.join(anno_rpath, 'vidvrd-baseline-output/short-term-predication.json')
short_term_predication_path = os.path.join(anno_rpath, 'vidvrd-baseline-output/results_re4mat.json')


def load_object_trajectory_proposal():
    """
    Test loading precomputed object trajectory proposals
    """
    dataset = VidVRD(anno_rpath=anno_rpath,
                     video_rpath=video_rpath,
                     splits=splits)

    video_indices = dataset.get_index(split='train')
    for vid in video_indices:
        durations = set(rel_inst['duration'] for rel_inst in dataset.get_relation_insts(vid, no_traj=True))
        for duration in durations:
            segs = segment_video(*duration)
            for fstart, fend in segs:
                trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=False, verbose=True)
                trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=True, verbose=True)

    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=False, verbose=True)
            trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=True, verbose=True)


def load_relation_feature():
    """
    Test loading precomputed relation features
    """
    dataset = VidVRD(anno_rpath=anno_rpath,
                     video_rpath=video_rpath,
                     splits=splits)

    extractor = feature.FeatureExtractor(dataset, prefetch_count=0)

    video_indices = dataset.get_index(split='train')
    for vid in video_indices:
        durations = set(rel_inst['duration'] for rel_inst in dataset.get_relation_insts(vid, no_traj=True))
        for duration in durations:
            segs = segment_video(*duration)
            for fstart, fend in segs:
                extractor.extract_feature(dataset, vid, fstart, fend, verbose=True)

    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            extractor.extract_feature(dataset, vid, fstart, fend, verbose=True)


def train():
    dataset = VidVRD(anno_rpath=anno_rpath,
                     video_rpath=video_rpath,
                     splits=splits)
    param = dict()
    param['model_name'] = 'baseline'
    param['rng_seed'] = 1701
    param['max_sampling_in_batch'] = 32
    param['batch_size'] = 64
    param['learning_rate'] = 0.001
    param['weight_decay'] = 0.0
    param['max_iter'] = 5000
    param['display_freq'] = 1
    param['save_freq'] = 5000
    param['epsilon'] = 1e-8
    param['pair_topk'] = 20
    param['seg_topk'] = 200
    print(param)

    model.train(dataset, param)


def detect(re_detect=True):
    dataset = VidVRD(anno_rpath=anno_rpath,
                     video_rpath=video_rpath,
                     splits=splits)
    with open(os.path.join(get_model_path(), 'baseline_setting.json'), 'r') as fin:
        param = json.load(fin)

    if re_detect:
        short_term_relations = model.predict(dataset, param)

        # save short-term predication results
        # with open(short_term_predication_path[:-5] + '.pkl', 'wb+') as stp_pkl_out_f:
        #     pickle.dump(short_term_relations, stp_pkl_out_f, protocol=pickle.HIGHEST_PROTOCOL)
        # print("Successfully save short-term predication to: " + short_term_predication_path[:-5] + '.pkl')

        with open(short_term_predication_path, 'w+') as stp_out_f:
            stp_out_f.write(json.dumps(short_term_relations))
        print("Successfully save short-term predication to: " + short_term_predication_path)

    else:
        # load short_term_relations from save file
        # with open(short_term_predication_path[:-5] + '.pkl', 'rb') as stp_pkl_in_f:
        #     short_term_relations = pickle.load(stp_pkl_in_f)

        with open(short_term_predication_path, 'r') as stp_in_f:
            short_term_relations = json.load(stp_in_f)

    print('greedy relational association ...')
    video_relations = dict()
    for vid in tqdm(short_term_relations.keys()):
        res = association.greedy_relational_association(short_term_relations[vid], param['seg_topk'])
        res = sorted(res, key=lambda r: r['score'], reverse=True)[:param['video_topk']]
        video_relations[vid] = res
    # save detection result
    with open(os.path.join(get_model_path(), 'baseline_relation_prediction.json'), 'w') as fout:
        output = {
            'version': 'VERSION 1.0',
            'results': video_relations
        }
        json.dump(output, fout)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='VidVRD baseline')
    # parser.add_argument('--load_feature', action="store_true", default=False, help='Test loading precomputed features')
    # parser.add_argument('--train', action="store_true", default=False, help='Train model')
    # parser.add_argument('--detect', action="store_true", default=False, help='Detect video visual relation')
    # args = parser.parse_args()
    #
    # if args.load_feature or args.train or args.detect:
    #     if args.load_feature:
    #         load_object_trajectory_proposal()
    #         load_relation_feature()
    #     if args.train:
    #         train()
    #     if args.detect:
    #         detect()
    # else:
    #     parser.print_help()

    # could run directly on Pycharm:
    # train()
    detect(False)

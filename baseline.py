import json
import os
from collections import defaultdict
import pickle

from tqdm import tqdm

from baseline import segment_video, get_model_path
from baseline import trajectory, feature, model, association
from dataset import VidVRD

# could modify these paths
anno_rpath = 'baseline/vidvrd-dataset'
video_rpath = 'baseline/vidvrd-dataset/videos'
splits = ['train', 'test']
short_term_predication_path = os.path.join(anno_rpath, 'vidvrd-baseline-output/short-term-predication.json')


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

    if re_detect:
        with open(os.path.join(get_model_path(), 'baseline_setting.json'), 'r') as fin:
            param = json.load(fin)
        short_term_relations = model.predict(dataset, param)

        # save short-term predication results
        with open(short_term_predication_path[:-5] + '.pkl', 'wb+') as stp_pkl_out_f:
            pickle.dump(short_term_relations, stp_pkl_out_f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Successfully save short-term predication to: " + short_term_predication_path[:-5] + '.pkl')

        # trans all of keys 2 str
        for each_key in short_term_relations.keys():
            short_term_relations['{}::{}::{}'.format(*each_key)] = short_term_relations[each_key]
            del short_term_relations[each_key]
        with open(short_term_predication_path, 'w+') as stp_out_f:
            stp_out_f.write(json.dumps(short_term_relations))
        print("Successfully save short-term predication to: " + short_term_predication_path)

    else:
        # load short_term_relations from save file
        # with open(short_term_predication_path, 'r') as stp_in_f:
        #     short_term_relations = json.load(stp_in_f)
        with open(short_term_predication_path[:-5] + '.pkl', 'rb') as stp_pkl_in_f:
            short_term_relations = pickle.load(stp_pkl_in_f)
            # for each_st in short_term_relations.items():
            #     # print(each_st[0])
            #     print(each_st[1][0])
            #     print(len(each_st[1][1]))
            #     print(each_st[1][2])

            # for each_key in list(short_term_relations.keys()):
            #     short_term_relations['{}-{}-{}'.format(*each_key)] = short_term_relations[each_key]
            #     del short_term_relations[each_key]
            # with open(short_term_predication_path, 'w+') as stp_out_f:
            #     json.dump(short_term_relations, stp_out_f)
            # print("Successfully save short-term predication to: " + short_term_predication_path)


    # group short term relations by video
    video_st_relations = defaultdict(list)
    for index, st_rel in short_term_relations.items():
        vid = index[0]
        video_st_relations[vid].append((index, st_rel))
    # video-level visual relation detection by relational association
    print('greedy relational association ...')
    video_relations = dict()
    for vid in tqdm(video_st_relations.keys()):
        video_relations[vid] = association.greedy_relational_association(
            dataset, video_st_relations[vid], max_traj_num_in_clip=100)
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

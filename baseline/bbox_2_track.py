import dlib
from dlib import drectangle
import numpy as np
import os
import pickle as pkl
from collections import defaultdict
import matplotlib

matplotlib.use('Agg')
from skimage import io
from dataset import get_dataset
from baseline.trajectory import Trajectory, traj_iou
from utils import create_video, profile

# from IPython import embed

gpu_fr_det_path = '/storage/dldi/PyProjects/FasterRCNN4VidVRDT1/data/output/res101/vidor/faster_rcnn_1_4_283995.pth'

_fr_det_root = '/home/xdshang/workspace/mm17/py_faster_rcnn/output/faster_rcnn_end2end/imagenet-vid_2016_relation1/resnet101_faster_rcnn_bn_scale_merged_end2end_2nd_iter_80000'


def cubic_nms(trajs, thresh):
    if len(trajs) == 0:
        return []
    iou = traj_iou(trajs, trajs)
    # get the order
    scores = [traj.score for traj in trajs]
    order = np.argsort(scores)[::-1]
    # nms
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inds = np.where(iou[i, order] <= thresh)[0]
        order = order[inds]
    return keep


@profile
def video_object_detection(dataset, vid, fstart, fend, max_tracking_num=20,
                           conf_thres=0.1, cls_nms_thres=0.3, all_nms_thres=0.5, vis=False):
    vsig = dataset.get_video_signature(vid, fstart, fend)
    print('-----Video Object Detection for {}-----'.format(vsig))
    # load per-frame detection results
    cls_trajs = defaultdict(list)
    cnt = 0
    for fid in range(fstart, fend):
        with open(os.path.join(_fr_det_root,
                               dataset.get_frame_index(vid, fid)) + '.pkl', 'r') as fin:
            dets = pkl.load(fin)
        for j, det in enumerate(dets):
            if j == 0:
                continue  # skip background category
            for i in range(det.shape[0]):
                cls_trajs[j - 1].append(Trajectory(fid - fstart,
                                                   drectangle(*det[i, :4].astype('float64')),
                                                   det[i, 4], det[i, 5:], j - 1, vsig))
                cnt += 1
    print('total # of per-frame detections is {}'.format(cnt))
    # load frames
    frames = []
    for fid in range(fstart, fend):
        img = io.imread(dataset.get_frame_path(vid, fid))
        frames.append(img)
    # track per-frame detections and nms
    all_trajs = []
    for j in cls_trajs.keys():
        trajs = cls_trajs[j]
        trajs.sort(reverse=True)
        trajs = [traj for traj in trajs[:max_tracking_num] if traj.score > conf_thres]
        # tracking
        for traj in trajs:
            anchor = traj.pstart
            # backward tracking
            tracker = dlib.correlation_tracker()
            tracker.start_track(frames[anchor], traj.roi_at(anchor))
            for fid in range(anchor - 1, -1, -1):
                tracker.update(frames[fid])
                roi = tracker.get_position()
                traj.predict(roi, reverse=True)
            # forward tracking
            tracker = dlib.correlation_tracker()
            tracker.start_track(frames[anchor], traj.roi_at(anchor))
            for fid in range(anchor + 1, len(frames)):
                tracker.update(frames[fid])
                roi = tracker.get_position()
                traj.predict(roi)
        keep = cubic_nms(trajs, cls_nms_thres)
        trajs = [trajs[i] for i in keep]
        print('\t# of video detections for {} is {}'. \
              format(dataset.get_class_name(j), len(trajs)))
        cls_trajs[j] = trajs
        all_trajs.extend(trajs)
    # overall nms
    if len(all_trajs) > max_tracking_num:
        keep = cubic_nms(all_trajs, all_nms_thres)
        all_trajs = [all_trajs[i] for i in keep]
    print('total # of video detections is {}'.format(len(all_trajs)))
    if vis:
        for traj in all_trajs:
            frames = traj.draw_on(dataset, frames)
        size = frames[0].shape[1], frames[0].shape[0]
        save_path = dataset.get_visualization_path('vot', vid)
        create_video(os.path.join(save_path, vsig), frames, 10, size, True)
    return all_trajs


@profile
def video_object_detection_gt(dataset, vid, fstart, fend, vis=False):
    vsig = dataset.get_video_signature(vid, fstart, fend)
    print('-----Video Object Detection GT for {}-----'.format(vsig))
    # load gt trajectories
    all_trajs = []
    anno = dataset.get_annotation(vid, fstart, fend)
    for trackid in anno['valid_objects']:
        traj = None
        for i, frame in enumerate(anno['frames']):
            cnt = 0
            for obj in frame['objects']:
                if obj['trackid'] == trackid:
                    roi = drectangle(float(obj['xmin']), float(obj['ymin']),
                                     float(obj['xmax']), float(obj['ymax']))
                    if traj is None:
                        traj = Trajectory(i, roi, 1., None,
                                          dataset.get_class_id(obj['name']), vsig, trackid)
                    else:
                        traj.predict(roi)
                    cnt += 1
            assert cnt == 1, 'Object {} should appear only once in each frame.'. \
                format(trackid)
        assert not traj is None, 'Object {} could not be found'.format(trackid)
        all_trajs.append(traj)
    # compute classemes
    for fid in range(fstart, fend):
        with open(os.path.join(_fr_det_root,
                               dataset.get_frame_index(vid, fid)) + '_norpn.pkl', 'r') as fin:
            dets = pkl.load(fin)
        for traj in all_trajs:
            cnt = 0
            det = dets[traj.category + 1]
            for i in range(det.shape[0]):
                roi = traj.roi_at(fid - fstart)
                if tuple(det[i, :4]) == (roi.left(), roi.top(), roi.right(), roi.bottom()):
                    if traj.classeme is None:
                        traj.classeme = det[i, 5:]
                    else:
                        traj.classeme += det[i, 5:]
                    cnt += 1
            assert not cnt == 0, 'No object detection found at {}.'.format(fid)
            assert not cnt > 1, 'Multiple object detections found at {}.'.format(fid)
    # normalize classemes
    for traj in all_trajs:
        traj.classeme = traj.classeme / (fend - fstart)
    print('total # of video detections is {}'.format(len(all_trajs)))
    if vis:
        # load frames
        frames = []
        for fid in range(fstart, fend):
            img = io.imread(dataset.get_frame_path(vid, fid))
            frames.append(img)
        for traj in all_trajs:
            frames = traj.draw_on(dataset, frames)
        size = frames[0].shape[1], frames[0].shape[0]
        save_path = dataset.get_visualization_path('vot_gt', vid)
        create_video(os.path.join(save_path, vsig), frames, 10, size, True)
    return all_trajs


def extract_trajectories(dataset, vid, fstart, fend, gt=False):
    vsig = dataset.get_video_signature(vid, fstart, fend)
    name = 'traj_cls_gt' if gt else 'traj_cls'
    path = dataset.get_feature_path(name, vid)
    path = os.path.join(path, '{}-{}.pkl'.format(vsig, name))
    if os.path.exists(path):
        # print('loading existing {}-{}.pkl'.format(vsig, name))
        with open(path, 'r') as fin:
            trajs = pkl.load(fin)
    else:
        if gt:
            trajs = video_object_detection_gt(dataset, vid, fstart, fend, vis=True)
        else:
            trajs = video_object_detection(dataset, vid, fstart, fend, max_tracking_num=20,
                                           conf_thres=0.1, cls_nms_thres=0.3, all_nms_thres=0.5, vis=True)
        with open(path, 'w') as fout:
            pkl.dump(trajs, fout)
    return trajs


def _sanity_check_gt(dataset, vid, fstart, fend, trajs):
    trakids = set(traj.gt_trackid for traj in trajs)
    anno = dataset.get_annotation(vid, fstart, fend)
    anno_trackids = set(anno['valid_objects'])
    if trakids != anno_trackids:
        print('--WARNING: {} not match ground-truth'.format(
            dataset.get_video_signature(vid, fstart, fend)))


if __name__ == '__main__':
    dataset = get_dataset('07-30')
    sample_index, _ = dataset.get_data(split='train+test', shuffle=False)
    # NOTE: better generate non_gt and gt separately
    for vid, fstart, fend in sample_index:
        trajs = extract_trajectories(dataset, vid, fstart, fend)
    # generatint gt may have assertion fails and warnings
    for vid, fstart, fend in sample_index:
        trajs = extract_trajectories(dataset, vid, fstart, fend, gt=True)
        _sanity_check_gt(dataset, vid, fstart, fend, trajs)

import os

# Several module level utility functions

root_path = '/home/daivd/PycharmProjects/VidVRD-py3/'
vidvrd_baseline_output_path = os.path.join(root_path, 'baseline/vidvrd-dataset/vidvrd-baseline-output')


def get_segment_signature(vid, fstart, fend):
    """
    Generating video clip signature string
    """
    return '{}-{:04d}-{:04d}'.format(vid, fstart, fend)


def get_feature_path(name, vid):
    """
    Path to save intermediate object trajectory proposals and features
    """
    path = os.path.join(vidvrd_baseline_output_path, 'features', name)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, vid)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_model_path():
    """
    Path to save trained model
    """
    path = os.path.join(vidvrd_baseline_output_path, 'models')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def segment_video(fstart, fend):
    """
    Given the duration [fstart, fend] of a video, segment the duration
    into many 30-frame segments with overlapping of 15 frames
    """
    segs = [(i, i + 30) for i in range(fstart, fend - 30 + 1, 15)]
    return segs

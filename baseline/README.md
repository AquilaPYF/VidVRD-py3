The baseline code for the VidVRD dataset introduced in the following paper.
```
@inproceedings{shang2017video,
    author={Shang, Xindi and Ren, Tongwei and Guo, Jingfan and Zhang, Hanwang and Chua, Tat-Seng},
    title={Video Visual Relation Detection},
    booktitle={ACM International Conference on Multimedia},
    address={Mountain View, CA USA},
    month={October},
    year={2017}
}
```

### Branch Python 3.6
```bash
conda create -n tensorflow pip python=3.6

source activate tensorflow

pip install --ignore-installed --upgrade 	https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl

pip install -r requirements.txt

python baseline.py
```
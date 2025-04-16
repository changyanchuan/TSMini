# TSMini - K Nearest Neighbor-Guided Trajectory Similarity Learning

The offcial pytorch implementation of TSMini.
```
@article{chang2025k,
  title={K Nearest Neighbor-Guided Trajectory Similarity Learning},
  author={Chang, Yanchuan and Cai, Xu and Jensen, Christian S and Qi, Jianzhong},
  journal={arXiv preprint arXiv:2502.00285},
  year={2025}
}
```

## Requirements
- A linux server with Python 3.11.0
- `pip install -r requirements.txt`
- Datasets can be downloaded from [here](https://drive.google.com/file/d/1JV1MinwlMv-0d27hUynVlHsAgOx2NGNX/view?usp=sharing). (Notes: the file is shared by an anonymous account and the link is untrackable and bi-anonymous.)
- `tar -zxvf TSMini_datasets.tar.gz -C ./data/` 


## Quick Start
Train and test TSMini to approximate Frechet on the Porto dataset:

```bash
python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name frechet
```

# TSMini

This is a pytorch implementation for the manuscript #XXX submitted to KDD25.

## Requirements
- A linux server with Python 3.11.0
- `pip install -r requirements.txt`
- Datasets can be downloaded from [here](https://drive.google.com/drive/folders/1wvFSdi4T1RvG1ww7TlobQJoTSBdJ7zWq?usp=sharing). (Notes: the link is untrackable and bianonymous.)
- `tar -zxvf TSMini_datasets_KDD.tar.gz -C ./data/` 


## Quick Start
Train and test TSMini to approximate Frechet on the Porto dataset:

```bash
python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name frechet
```

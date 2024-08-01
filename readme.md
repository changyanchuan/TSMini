# TSMini

This is a pytorch implementation for the manuscript #870 submitted to KDD25.

## Requirements
- A linux server with Python 3.11.0
- `pip install -r requirements.txt`
- Datasets can be downloaded from [here](https://drive.google.com/drive/folders/1ee7i7TkqgBXqHUlMzyhnRFddlYwbmOYy?usp=sharing). (Notes: the file is shared by an anonymous account and the link is untrackable and bi-anonymous.)
- `tar -zxvf TSMini_datasets_KDD.tar.gz -C ./data/` 


## Quick Start
Train and test TSMini to approximate Frechet on the Porto dataset:

```bash
python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name frechet
```

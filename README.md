# Motron
![](./hero_figure.png)
Hi, and welcome to the codebase for our CVPR22 submission.
Feel free to explore one of our pre-trained models using one of the notebooks in the "notebooks" folder.
For example re-run the evaluation on H3.6M in the "RES H3.6M Deterministic Evaluation.ipynb" notebook.

Or train your own model via

    PYTHONPATH=. python -O h36m/train.py --device cpu --config ./config/h36m.yaml

Model configuration is passed via the YAML file located in the "config" folder.

## Install Instructions

Please download the processed H3.6M dataset from [here](https://c.web.de/@1042139905007819411/Skz7YT_ZRwiPSnxDgCg_eQ) and place it in
    
    ./data/processed/h3.6m.npz

Install the requirements

    pip install -r requirements.txt

If your device supports GPU acceleration please uncomment line 5 in requirements.txt.

To visualize AMASS output please additionally install

    git+https://github.com/nghorbani/configer@5f2b82e5
    git+https://github.com/nghorbani/human_body_prior@1936f38
    git+https://github.com/nghorbani/amass@e5f0889

The main results can be found [here](https://github.com/motron-cvpr22/motron/blob/master/notebooks/RES%20Eval%20Gen.ipynb) and [here](https://github.com/motron-cvpr22/motron/blob/master/notebooks/RES%20H3.6M%20Deterministic%20Evaluation.ipynb).

# Preprocess Data

## H36M
To download and preprocess the H36M dataset automatically run 

     PYTHONPATH=. python ./preprocess/h36m.py

## AMASS
Download the AMASS datasets you want to use from https://amass.is.tue.mpg.de

To preprocess run

     PYTHONPATH=. python ./preprocess/amass.py --path [path to amass datasets] --out ./data/processed/amass --datasets [List of datasets to process]

In the paper we eventually also follow the amass evaluation procedure of [Aksan et al.](https://github.com/eth-ait/motion-transformer).
For the future we suggest to use their evaluation data/split as the distribution difference between train and test data is more reasonable.

# Cite Us
```
@article{salzmann2022motron,
  title={Motron: Multimodal Probabilistic Human Motion Forecasting},
  author={Salzmann, Tim and Pavone, Marco and Ryll, Markus},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

If you use data from AMASS or H36M, please cite the original papers as detailed on their website.
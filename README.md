<div align="center">

# MANNeR

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

A NEW VERSION OF THE FRAMEWORK IS AVAILABLE IN [NewsRecLib](https://github.com/andreeaiana/newsreclib)

## Description

This is the implementation of the MANNeR framework from the paper "Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation"

![](./framework.png)

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/andreeaiana/manner
cd manner

# [OPTIONAL] create conda environment
conda create -n manner_env python=3.9
conda activate manner_env

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64

# train on GPU
python src/train.py trainer=gpu
```

Run ensemble (CR-Module + A-Module) with trained sub-modules with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml train=False
```

## Citation

```bibtex
@misc{iana2023train,
      title={Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation}, 
      author={Andreea Iana and Goran Glavaš and Heiko Paulheim},
      year={2023},
      eprint={2307.16089},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
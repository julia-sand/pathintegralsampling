# [Path Integral Sampler: a stochastic control approach for sampling](https://arxiv.org/abs/2111.15141)
some notes: The code is adapted so that it will run on later versions of pytorch lightning. Also removed logger tools that required signup (eg wandb), additional files and configured it for the OU process experiment. changed the dataloader so that can also now be run on CPU during testing.

## Setup

The repo heavily depends on [jam](https://github.com/qsh-zh/jam), a versatile toolbox developed by [Qsh.zh](https://github.com/qsh-zh) and [jam-hlt](https://github.com/qsh-zh/jam), a decent deep leanring project template. [⭐️](https://github.com/qsh-zh/jam) if you like them.

*[poetry](https://python-poetry.org/)* (**Recommended**)
```shell
curl -fsS -o /tmp/get-poetry.py https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py
python3 /tmp/get-poetry.py -y --no-modify-path
export PATH=$HOME/.poetry/bin:$PATH
poetry shell
poetry install
```

*pip*
```shell
pip install .
```
Note: when I installed the package jam manually using pip, I also needed to add the file meta.py (definition of as_numpy function) 


## Reproduce

```
python run.py experiment=ou.yaml logger=csv
```

See the [folder](configs/experiment) for more experiment configs.

There are some [results](https://wandb.ai/qinsheng/pub_pis?workspace=user-qinsheng) reproduced by the repo.

## Reference

```tex
@inproceedings{zhang2021path,
  author    = {Qinsheng Zhang and Yongxin Chen},
  title     = {Path Integral Sampler: a stochastic control approach for sampling},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

## MICS:

- [sde-sampler](https://gitlab.com/qsh.zh/sde-sampler/-/tree/rings) Uncleaned code used for experiments in paper.
- SDE parameters `dt,g` are modified due to [the issue](https://github.com/google-research/torchsde/issues/109).

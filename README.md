# Dreamer-CDP: Improving Reconstruction-free World Models Via Continuous Deterministic Representation Prediction

We first thank Danijar Hafner for the release of [DreamerV3](https://github.com/danijar/dreamerv3/tree/main).


## Dreamer-CDP

Dreamer-CDP learns a world model without reconstruction through continuous deterministic representation prediction. It reaches similar performance as the reconstruction-based Dreamer-V3 on the Crafter environment. Link to the paper:

<p align="center">
  <a href="https://arxiv.org/abs/2603.07083"><img src="https://img.shields.io/badge/arXiv-2603.07083-b31b1b.svg" alt="arXiv" /></a>
</p>


# Instructions

The code has been tested on Linux and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX](https://github.com/jax-ml/jax#pip-installation-gpu-cuda) and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

If you find this code useful, please reference in your paper:

```
@misc{hauri2026dreamercdp,
      title={Dreamer-CDP: Improving Reconstruction-free World Models Via Continuous Deterministic Representation Prediction}, 
      author={Michael Hauri and Friedemann Zenke},
      journal={arXiv preprint arXiv:2603.07083}
}
```

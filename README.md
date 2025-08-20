# pytorch-a3c-gymnasium

This is a PyTorch implementation of Asynchronous Advantage Actor Critic (A3C) from ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf).

This implementation is inspired by [Universe Starter Agent](https://github.com/openai/universe-starter-agent).
In contrast to the starter agent, it uses an optimizer with shared statistics as in the original paper.

## Install
```bash
pip install gymnasium
pip install numpy
pip install torch
pip install opencv-python
```

## Usage
```bash
# Works only wih Python 3.
python3 main.py --num-processes 16
```

This code runs evaluation in a separate thread in addition to 16 processes.

## Results
<TBD>

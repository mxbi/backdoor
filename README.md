# Backdoors in Neural Networks

Code for my undergraduate 3rd year project (as well as my Architectural Backdoors in Neural Networks paper: https://arxiv.org/abs/2206.07840).

The `backdoor` library can be found in the `backdoor/` folder.

Installation is easy, just `pip install .`.

- For architectural backdoors, see `backdoors/models.py`.
- For a reimplementation of Handcrafted Backdoors in Deep Neural Networks (Hong et al.) see `models/handcrafted.py`.

Everything in this repo is released under MIT license, unless specified otherwise.

### Setting up for development

```bash
conda create -f environment.yml # Anaconda environment used for development
pip install -e . # Editable install of the backdoor library
```

### Running tests

```
pytest --cov=backdoor tests/
```

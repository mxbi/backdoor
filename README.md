# Backdoors in Neural Networks

The `backdoor` library can be found in the `backdoor/` folder.

Installation is easy, just `pip install .`.

`pt.darts/` contains a modified version of https://github.com/khanrc/pt.darts to insert architectural backdoors (most code in that folder is not mine).

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
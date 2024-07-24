## Intro

contains EMG TimeSeries data analysis notebooks

## Project index
- [Naruto](./emg_analysis/naruto_emg.ipynb)

## Installs

For Mac with homebrew installed (get homebrew if you own a Mac):

```bash
brew install pipx
pipx ensure path
pipx install poetry
```

For windows/linux

```bash
# Recommended: install scoop
scoop install pipx
pipx ensurepath

# If you installed python using Microsoft Store, replace `py` with `python3` in the next line.
py -m pip install --user pipx
pipx ensurepath
```

## Setup

**Recommended VS extension for .ipynb files:** get [Jupyter notebook extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
I've setup poetry to install a virtualenv with all the necessary data science tools installed.

```bash
# in Biosignal-Preprocessing/emg-analysis
cd ./emg_analysis
poetry install
poetry run
```

For running the notebooks with the poetry environment follow this [guide](https://maeda.pm/2024/03/03/python-poetry-and-vs-code-2024/)

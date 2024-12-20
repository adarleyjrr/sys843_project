# Final Project Scripts - SYS843

Note: The scripts used in this project are only designed to work with GPUs if Apple Silicon is present.

## Virtual environment

Use `conda` to create new virtual environment:

```
conda create --name sys843_project
```

Activate virtual environment:

```
conda activate sys843_project
```

Install Python at version 3.12 to ensure compability with Pytorch libraries:

```
conda install python=3.12
```

Next, install Pythorch libraries:

```
pip install torch torchvision
```

Then install timm (library that has various pre-trained models):

```
pip install timm
```

Then install `matplotlib` to plot graphs:

```
pip install matplotlib
```

Follow instalation of `coremltools` as described in [Installing Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/installing-coremltools.html) guide:

```
pip install coremltools
```

## Hyperparameter tuning script

You can run the Python file `train_hyperparameter_tuning.py` under the virtual environment by making sure the environment is active and then:

```
python train_hyperparameter_tuning.py --model fastvit_t8
```

To find a set of optiomal hyperparameters for FastViT-T8.

## Train script script

You can run the Python file `train.py` under the virtual environment by making sure the environment is active and then:

```
python train.py --model fastvit_t8
```

To train FastViT-T8.

## Plot script script

You can run the Python file `train_plots.py` under the virtual environment by making sure the environment is active and then:

```
python train_plots.py
```

## Convert model to Core ML script

You can run the Python file `convert_coreml.py` under the virtual environment by making sure the environment is active and then:

```
python convert_coreml.py --model fastvit_t8
```

To convert the model FastViT-T8 to Core ML.

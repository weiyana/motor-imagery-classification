# Coarse-grained and Fine-level Feature Learning for Upper-limb Related Motor Imagery Classification

## 1. Installation

### Environment

- Python == 3.7.10
- PyTorch == 1.9.0
- CUDA == 10.2

### Dependencies

**Create conda environment**



Install packages manually

```shell
conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install numpy pandas matplotlib pyyaml ipywidgets
pip install torchinfo braindecode moabb
```

## 2. Directory structure

```
.
├── README.md
├── base
│   └── base_trainer.py
├── bci-2021.yaml
├── configs
│   └── bci2021_config.yaml
├── data
│   ├── Train
│   ├── Test
│   ├── Valid
├── data_loader
│   ├── __pycache__
│   ├── data_generator.py
│   ├── dataset
│   └── preprocessor.py
├── figures
│   └── figure.png
├── history.ipynb
├── main.py
├── models
│   ├── __pycache__
│   ├── bci2021_model.py
│   └── model_builder.py
├── runs
│   ├── train_all_subject.sh
│   └── train_single_subject.sh
├── trainers
│   ├── __pycache__
│   ├── bci2021_trainer.py
│   └── trainer_maker.py
└── utils
├── calculator.py
├── get_args.py
├── logger.py
└── utils.py
```

<!-- ## 4. Dataset

- Use [braindecode](https://braindecode.org)

**BCI Competition IV-2a dataset**

- 9 subjects
- Classes: left hand, right hand (2 classes)
- Session-to-session set up (=subject dependent)
- Training set: 144 trials per subject
- Test set: 144 trials per subject

**Preprocessing**

- Sampling rate: 250Hz
- Time segment: [-0.5, 4.0]s post-cue
- Band-pass filtering: 0-42Hz
- Normalization: exponential moving average

## 5. Experiments

|Models|S01|S02|S03|S04|S05|S06|S07|S08|S09|Mean|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
BCI-2021|97.92|71.53|97.22|84.72|72.92|74.31|99.31|84.03|97.22|86.58| -->

## 6. Get Started
**Data preparation**

```shell
cd motor-imagery-classification #project folder
ln -s $data_path data #
```

**Training all subjects**

```shell
sh runs/train_all_subject_grasp.sh
```

**Training single subject**

```shell
sh runs/train_single_subject.sh
```


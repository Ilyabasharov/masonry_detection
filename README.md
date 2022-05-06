# masonry_detection
Masonry detection via neural networks and classic cv algorithms

## Installation

You can use [venv](https://docs.python.org/3/tutorial/venv.html) to install libraries.

### Install requirements

```bash
pip install -r requirements.txt
```

### Download weights

```bash
cd scripts
sh download_data.sh
```

## How to run

### Train segmentation net

```bash
cd src
python3 segmentation/train.py
```

### Run masonry detection

```bash
cd src
python3 main.py
```
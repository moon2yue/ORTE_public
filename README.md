# ORTE_public

## Datasets and Running Environments

All data can be found from the manuscript reference. Here we take **freqshape** as an example and provide the raw data.


Requirements are found in `requirements.txt`. Please install the necessary requirements via pip (recommended) or conda.

## How to run

**Example**: Detailed examples of the model are given in the experiment scripts found in `experiments` directory. 
A simple example is given for the freqshape dataset in `experiments/freqshape`

1. Training a black-box model:

`python train_transformer.py`

2. Explain the black-box model:

`python bc_model_ptype.py`

3. Occlusion experiments of real-world datasets in `experiments/evaluation`:

`python occlusion_exp.py`

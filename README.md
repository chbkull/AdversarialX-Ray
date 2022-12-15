# AdversarialX-Ray
Study of adversarial attacks against X-ray analysis models

## Running Locally
This project was developed using Python 3.9.15 and CUDA 11.6. In order to run locally, take the following steps:
- (Optional) create a virtual environment: `python -m venv venv`
- Install all required packages: `python -m pip install -r python3.9-requirements.txt`
- Download the dataset: `bash download_data.sh`. Note that this will download and unpack all twelve tarballs, which will take quite a while.
- If you wish to train, run all cells in `train_model.ipynb`
- If you wish to evaluate, run all cells in `evaluate_model.ipynb`
- If you wish to test out adversarial attacks, run all cells in `test_attacks.ipynb`
- If you wish to inspect the layer activations of the model, run all cells in `inspect_layers.ipynb`
- If you wish to generate white-box adversarial examples, run `adversarial_generator.py`
- If you wish to generate black-box adversarial examples, run `black_box_zoo.py`
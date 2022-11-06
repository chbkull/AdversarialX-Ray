# AdversarialX-Ray
Study of adversarial attacks against X-ray analysis models

## Running Locally
This project was developed using Python 3.9.15 and CUDA 11.6. In order to run locally, take the following steps:
- (Optional) create a virtual environment: `python -m venv venv`
- Install all required packages: `python -m pip install -r python3.9-requirements.txt`
- Download the dataset: `sh download_data.sh`. Note that this will only download and unpack the first of the twelve tarballs for space reasons (just one comes out to ~2GB)
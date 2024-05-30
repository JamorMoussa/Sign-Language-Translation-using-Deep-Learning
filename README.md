# Sign-Language-Translation-using-Deep-Learning

**Motivation:** How Can we use AI technologie to help Deaf People?

This project aims to build a Sign Language Translator system, which able to map each sign into english lettre. So, we're going to test vaious Deep Learning architectures including **Multi-layer Perceptron (MLP)**, **Convolution Neural Networks (CNN)** and **Graph Convolution Networks (GCN)**. 


## Project setup

Let's first create a virtual enviroment for this project

```bash
python3 -m venv --system-site-packages env
```

Next, install the `asl` library, which contains datasets for various architectures mentionned previously.

```bash
pip install -e .
```

**Note:** before install the `asl` lib, make sure that the `env` has been activated.

## Data Processing

For data processing you need to run the `process_data.py` script, with different arguments.

```bash
python3 process_data.py --which=gcn --max_sample=100 --replace=true
```
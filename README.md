# Sign-Language-Translation-using-Deep-Learning

**Motivation:** How Can we use AI technologie to help Deaf People?

This projects aims to build a Sign Language Translator system, which able to map each sign into english lettre. In this projects we're going to test vaious Deep Learning architectures including **Multi-layer Perceptron (MLP)**, **Convolution Neural Networks (CNN)** and **Graph Convolution Networks (GCN)**  


## Project setup

Let's first create a virtual enviroment for this projects

```bash
python3 -m venv --system-site-packages env
```

Next, install the `asl` library which contains datasets for various architectures mentionned previously.

```bash
pip install -e .
```

**Note:** before install the `asl` lib, insure that the `env` is has been activated.
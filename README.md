# QuantLib
**QuantLib** is a library to train deploy quantised neural networks (QNNs).
It was developed on top of the  [PyTorch](https://pytorch.org/) deep learning framework.

QuantLib is a component of QuantLab, which also includes **organising software** to manage machine learning (ML) experiments (`systems` and `manager` packages, as well as the `main.py` façade script).

## Installation and usage

### Create an Anaconda environment and install `quantlib`

Use [Anaconda](https://docs.anaconda.com/anaconda/install/) or Miniconda to install QuantLab's prerequisites.
You can find a `quantlab.yml` at https://github.com/pulp-platform/quantlab/blob/main/quantlab.yml 
```
$ conda env create -f quantlab.yml
```

After creating the conda environment, install the `quantlib` quantisation library in your Anaconda environment:
```
$ conda activate quantlab
(quantlab) $ cd quantlib
(quantlab) $ python setup.py install
(quantlab) $ cd ..
```

### Usage examples
- [MNIST example (first usage)](examples/mnist/MNIST_Example.ipynb)

## Notice

### Licensing information
`quantlib` is distributed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

In case you are planning to use QuantLab and `quantlib` in your projects, you might also want to consider the licenses under which the packages on which they depend are distributed:

* PyTorch - a [mix of licenses](https://github.com/pytorch/pytorch/blob/master/NOTICE), including the Apache 2.0 License and the 3-Clause BSD License;
* TensorBoard - [Apache 2.0 License](https://github.com/tensorflow/tensorboard/blob/master/LICENSE);
* NetworkX - [3-Clause BSD License](https://github.com/networkx/networkx/blob/main/LICENSE.txt);
* GraphViz - [MIT License](https://github.com/graphp/graphviz/blob/master/LICENSE);
* matplotlib - a [custom license](https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE);
* NumPy - [3-Clause BSD License](https://github.com/numpy/numpy/blob/main/LICENSE.txt);
* SciPy - [3-Clause BSD License](https://github.com/scipy/scipy/blob/master/LICENSE.txt);
* Mako - [MIT License](https://github.com/sqlalchemy/mako/blob/master/LICENSE);
* Jupyter - [3-Clause BSD License](https://github.com/jupyter/notebook/blob/master/LICENSE).

### Authors
* Matteo Spallanzani <<a href="mailto:spmatteo@iis.ee.ethz.ch">spmatteo@iis.ee.ethz.ch</a>> (ETH Zurich, now at Axelera AI)
* Georg Rutishauser  <<a href="mailto:georgr@iis.ee.ethz.ch">georgr@iis.ee.ethz.ch</a>> (ETH Zurich)
* Moritz Scherer     <<a href="mailto:scheremo@iis.ee.ethz.ch">scheremo@iis.ee.ethz.ch</a>> (ETH Zurich)
* Francesco Conti    <<a href="mailto:f.conti@unibo.it">f.conti@unibo.it</a>> (University of Bologna)

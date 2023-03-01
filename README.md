# QuantLib
**QuantLib** is a library to train deploy quantised neural networks (QNNs).
It was developed on top of the  [PyTorch](https://pytorch.org/) deep learning framework.

QuantLib is a component of QuantLab, which also includes **organising software** to manage machine learning (ML) experiments (`systems` and `manager` packages, as well as the `main.py` façade script).

## Installation and usage

### Create an Anaconda environment and install `quantlib`

Use [Anaconda](https://docs.anaconda.com/anaconda/install/) or Miniconda to install QuantLab's prerequisites.

**PyTorch 1.13.1 (Recommended)**
```sh
$> conda create --name pytorch-1.13
$> conda activate pytorch-1.13
$> conda config --env --add channels conda-forge
$> conda config --env --add channels pytorch 
$> conda install python=3.8 pytorch=1.13.1 pytorch-gpu torchvision=0.14.1 torchtext=0.14.1 torchaudio-0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
$> conda install ipython packaging parse setuptools tensorboard tqdm networkx python-graphviz scipy pandas ipdb onnx onnxruntime einops yapf tabulate
$> pip install setuptools==59.5.0 torchsummary parse coloredlogs netron
```

**PyTorch 1.12.1**
```sh
$> conda create --name pytorch-1.12
$> conda activate pytorch-1.12
$> conda config --env --add channels conda-forge
$> conda config --env --add channels pytorch 
$> conda install python=3.8 pytorch=1.12.1 torchvision=0.13.1 torchtext=0.13.1 torchaudio=0.12.1 -c pytorch -c conda-forge
$> conda install ipython packaging parse setuptools tensorboard tqdm networkx python-graphviz scipy pandas ipdb onnx onnxruntime einops yapf tabulate
$> pip install setuptools==59.5.0 torchsummary parse coloredlogs netron
```

After creating the conda environment, install the `quantlib` quantisation library in your Anaconda environment:
```
$ conda activate quantlab
(quantlab) $ cd quantlib
(quantlab) $ python setup.py install
(quantlab) $ cd ..
```

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
* Philip Wiese       <<a href="mailto:wiesep@ethz.ch">wiesep@ethz.ch</a>> (ETH Zurich)
* Francesco Conti    <<a href="mailto:f.conti@unibo.it">f.conti@unibo.it</a>> (University of Bologna)

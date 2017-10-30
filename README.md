# pix2pix-tensorflow

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)

Tensorflow implementation of pix2pix.  Learns a mapping from input images to output images, like these examples from the original paper:

<img src="docs/examples.jpg" width="900px"/>

This port is based directly on the torch implementation, and not on an existing Tensorflow implementation.  It is meant to be a faithful implementation of the original work and so does not add anything.  The processing speed on a GPU with cuDNN was equivalent to the Torch implementation in testing.

## Setup

### Prerequisites
- Tensorflow 1.0.0, Works with Tensorflow 1.3

### Recommended
- Tensorflow GPU edition + cuDNN

### Getting Started

```sh
Install Python 3. The version I used is Python 3.5.

If you are planning on using cuda with an nvidia GPU, :
Install cuDNN 6, and Cuda 8 according to https://developer.nvidia.com/cuda-toolkit.

Install tensorflow for python 3 with python -m pip install tensorflow-gpu 

If you are planning on using CPU:

Install tensorflow for python 3 with python -m pip install tensorflow

# clone this repo
git clone https://github.com/Rehket/pix2pix-tensorflow.git
cd pix2pix-tensorflow

Exported models can be placed in pix2pix-tensorflow/Models.

# Running:

python serve.py --path/to/models 

# will start a webserver were users can submit images to be processed.

```

TODO: Add the training instructions.



## Unimplemented Features

The following models have not been implemented:
- defineG_encoder_decoder
- defineG_unet_128
- defineD_pixelGAN

## Citation
If you use this code for your research, please cite the paper this code is based on: <a href="https://arxiv.org/pdf/1611.07004v1.pdf">Image-to-Image Translation Using Conditional Adversarial Networks</a>:

```

@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}

```

## Acknowledgments
This is a port of [pix2pix](https://github.com/phillipi/pix2pix) from Torch to Tensorflow.  It also contains colorspace conversion code ported from Torch.  Thanks to the Tensorflow team for making such a quality library!  And special thanks to Phillip Isola for answering my questions about the pix2pix code.

## LeNet

> [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)

## Abstract

<!-- [ABSTRACT] -->

Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns, such as handwritten characters, with minimal preprocessing. This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task. Convolutional neural networks, which are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques. Real-life document recognition systems are composed of multiple modules including field extraction, segmentation recognition, and language modeling. A new learning paradigm, called graph transformer networks (GTN), allows such multimodule systems to be trained globally using gradient-based methods so as to minimize an overall performance measure. Two systems for online handwriting recognition are described. Experiments demonstrate the advantage of global training, and the flexibility of graph transformer networks. A graph transformer network for reading a bank cheque is also described. It uses convolutional neural network character recognizers combined with global training techniques to provide record accuracy on business and personal cheques. It is deployed commercially and reads several million cheques per day..

<!-- [IMAGE] -->

## Results and models

### MNIST


| Backbone | Validation Accuracy | config | download    |
| :--------- | --------------------- | :------- | ------------- |
| LeNet5   | 98.97%              | config | model / log |

## Citation

```bibtext
@ARTICLE{726791,
  author={Lecun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
  journal={Proceedings of the IEEE}, 
  title={Gradient-based learning applied to document recognition}, 
  year={1998},
  volume={86},
  number={11},
  pages={2278-2324},
  doi={10.1109/5.726791}}
```

```bibtext
@ARTICLE{6296535,
  author={Deng, Li},
  journal={IEEE Signal Processing Magazine}, 
  title={The MNIST Database of Handwritten Digit Images for Machine Learning Research [Best of the Web]}, 
  year={2012},
  volume={29},
  number={6},
  pages={141-142},
  doi={10.1109/MSP.2012.2211477}}
```

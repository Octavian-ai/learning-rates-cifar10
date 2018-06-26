# CIFAR-10 learning rates 
This model will perform a grid search over 40 different learning rates and allows us to select the best LR for our target test accuracy. We have used different stopping conditions to save on the GPU time. 

## Usage

### Download code

```shell
git clone https://github.com/Octavian-ai/learning-rates-cifar10
```
### Creating virtualenv and running the code on local machine

```shell
virtualenv -p python3 envname
source envname/bin/activate
pip3 install numpy tensorflow
cd learning-rates-cifar10
python3 train-local.py
```
### Running the code on GPU
Training is quite slow without a GPU. It is easy to run on FloydHub. Instructions are given below.

```shell
sudo pip install -U floyd-cli
floyd login
cd learning-rates-cifar10
floyd init learning-rates-cifar10
floyd run --gpu --env tensorflow-1.8 --data signapoop/datasets/cifar-10/1:/data_set 'python train-floyd.py'
```
## Results
![](https://drive.google.com/uc?export=view&id=1u9k_Bx6fUbJtjX8b0P6tbiIrWzybwNHr)

## Acknowledgements

I would like to thank David Mack and Andrew Jefferson for their support on this task. 

The experimental setup of choosing best learning rate has been derived from David Mack's [article](https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2).

The model for the experiment is used from Serhiy Mytrovtsiy's work available on [GitHub](https://github.com/exelban/tensorflow-cifar-10).

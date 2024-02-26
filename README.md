# LDM-S in Keras
An implementation of LDM-based active learning algorithm referred to as LDM-Seeding (LDM-S). The LDM is a measure of the sample's closeness to the decision boundary based on the disagree metric in hypothesis space.

## Abstract
Active learning is a machine learning paradigm that aims to improve the performance of a model by strategically selecting and querying unlabeled data. One effective selection strategy is to base it on the model's predictive uncertainty, which can be interpreted as a measure of how informative a sample is. The sample's distance to the decision boundary is a natural measure of predictive uncertainty, but it is often intractable to compute, especially for complex decision boundaries formed in multiclass classification tasks.
To address this issue, this paper proposes the least disagree metric (LDM), defined as the smallest probability of disagreement of the predicted label, and an estimator for LDM proven to be asymptotically consistent under mild assumptions. The estimator is computationally efficient and can be easily implemented for deep learning models using parameter perturbation. The LDM-based active learning is performed by querying unlabeled data with the smallest LDM. Experimental results show that our LDM-based active learning algorithm obtains state-of-the-art overall performance on all considered datasets and deep architectures.

### Prerequisites:
- Linux
- Python 3.7
- NVIDIA GPU + CUDA 10.0, CuDNN 7.6

### Installation
The Keras and TensorFlow 2.0 can be installed using
```
pip3 install -r requirements.txt
```
The code requires NumPy, CuPy, Scikit-learn, SciPy, and tqdm libraries.

### Running an experiment
```
python3 run_mnist.py
```
runs an active learning experiment on MNIST dataset with S-CNN network, querying batches of 20 samples according to the LDM-S algorithm.

```
python3 run_cifar10.py
```
runs an active learning experiment on CIFAR10 dataset with K-CNN network, querying batches of 400 samples according to the LDM-S algorithm.

arguments:
```
--nBatch: batch size for training
--nEpoch: number of epochs for training
--nValid: number of samples for validation set
--nInit: number of initial labeled samples
--nPool: number of samples for pooling set
--nQuery: number of queries at each step
--nStep: number of acquisition steps
--nSample: number of samples for approximating rho
```

The results will be saved in `results/{dataset}_{network}/test_accs_{#rep}.txt`

## Contact
If there are any questions or concerns, feel free to send a message at ipcng00@gmail.com

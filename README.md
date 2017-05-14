# Investigating-Experimentally-Deep-Learning-without-Poor-Local-Minima-by-Kawaguchi

In a recent work by Kawaguchi (Kawaguchi, 2016), it as been claimed that under several assumptions met in practice, 
all local minima of the training loss function are equivalent feedforward models. An experimental assessment of such claims is
proposed by training pairs of identical networks with different weight initialization. In addition to comparing loss values,
several other metrics are considered in order to compare model similarity regarding label classifications and generalization.

Original paper : https://arxiv.org/abs/1605.07110

# Usage

jupyter notebook src/experiment.ipynb

Run all cells

Same for boston_experiment.ipynb or cifar10_experiment.ipynb

# Installation

$ pip install keras numpy scipy seaborn matplotlib

# Todo

- Manage the parameters
- Do experimentations with weight values which are near
- Add other datasets.
- Check the saddle points.
- Compare the same model with ADAM and SGD.
- Perturbate the weights after convergence.
- Define saddle points
-  Add a breakpoint to stop the experimentation. 
- Driver for cloud provider (AWS, AzureML)
- Store experimentations 

 
 # Class taught by  : 
 
 Antoine Cornuéjols : http://www.agroparistech.fr/ufr-info/membres/cornuejols/

 Class link : http://www.agroparistech.fr/ufr-info/membres/cornuejols/Teaching/Master-AIC/M2-AIC-advanced-ML.html


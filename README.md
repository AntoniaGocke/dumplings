# FeatureCloud App Federated Unlearning

This App can be ued with a CNN input from the fc-dep-learning App.

## Input parameter needed:
input_file: with the data which will be unlearned <br />
input_model: the global model will be used as a constraint for the gradient ascent which should enable a faster convergence when relearning <br />
### hyper_params: <br />
    delta: None
    tau: is the accuracy to which the model will be unlearned to, based on the Paper "Federated Unlearning: How to Efficiently Erase a Client in FL? (https://doi.org/10.48550/arXiv.2207.05521)" it will be set to 0.12 
### OptimizerValues <br />
    optName: 'AdamW' 
    opt_lr: 0.001 
    opt_weight_decay: 0.05 



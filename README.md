# FeatureCloud App Federated Unlearning

This App can be used with a CNN input from the fc-dep-learning App.

## Input parameter needed:
The following parameters are needed to be provided in the config - file.

### config.yaml <br />
input_file: 'train.npz' 
input_model: 'model.pt' 
output_file: 'model.pt' 
n_classes: 10 
in_features: 1 
n_clients: 4 
hyper_params: 
tau: 0.12 
OptimizerValues
opt_lr: 0.001 
opt_weight_decay: 0.05 

## Workflow

First you have to obtain a trained model, which will be used as an input for the App. Additionally, the target data to be forgotten will also ba used as an input for the model. Using a projected gradient ascent, the loss will be maximized. The unlearned model will then need to be retreined using the featurecloud deep learning app, where the model can be used as an input and be retrained with the remaining data.

# FeatureCloud App Federated Unlearning

This App can be used with a CNN input from the fc-dep-learning App.

## Input parameter needed:
The following parameters are needed to be provided in the config - file.

#### config.yaml
input_file: 'train.npz' <br />
input_model: 'model.pt' <br />
output_file: 'model.pt' <br />
n_classes: 10 <br />
in_features: 1 <br />
n_clients: 4 <br />
hyper_params: <br />
tau: 0.12 <br />
OptimizerValues <br />
opt_lr: 0.001 <br />
opt_weight_decay: 0.05 <br /> 

## Workflow

First you have to obtain a trained model, which will be used as an input for the App. Additionally, the target data to be forgotten will also ba used as an input for the model. Using a projected gradient ascent, the loss will be maximized. The unlearned model will then need to be retreined using the featurecloud deep learning app, where the model can be used as an input and be retrained with the remaining data.

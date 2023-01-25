from FeatureCloud.app.engine.app import AppState, app_state, Role
import bios
import numpy as np
import pandas as pd
from classes.CNN import CNN
from classes.UntrainFed import Gym, UnlearnGym, FederatedGym, FederatedUnlearnGym, ClientUnlearnGym
from sklearn.model_selection import train_test_split
from classes.CustomDataset import CustomDataSet
from torch import optim, load
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn import metrics


INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

INITIAL_STATE = 'initial'
COMPUTE_STATE = 'compute'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.


@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE)

    def run(self):
        self.log('Reading configuration file...')
        config = bios.read(f'{INPUT_DIR}/config.yaml')
        self.log(f'{config}')
        input_file = config['input_file']
        input_model = config['input_model']
        n_classes = config['n_classes']
        in_features = config['in_features']
        hyper_params = config['hyper_params']
        self.store('n_classes', n_classes)
        self.store('in_features', in_features)
        self.store('delta', hyper_params['delta'])
        self.store('tau', hyper_params['tau'])
        self.store('opt_lr', hyper_params['opt_lr'])
        self.store('opt_weight_decay', hyper_params['opt_weight_decay'])
        self.log('Done reading configuration.')

        # load npz file
        npz_file = np.load(f'{INPUT_DIR}/{input_file}', allow_pickle=True)

        # read images from npz
        self.log('Reading untraining images...')
        input_images = npz_file['data']
        self.store('input_images', input_images)

        #read labels from npz
        self.log('Reading untraining labels...')
        input_labels = npz_file['targets']
        self.store('input_labels', input_labels)

        # load model
        self.log('Reading model')
        model = CNN(n_classes=n_classes, in_features=in_features)
        model = load(f'{INPUT_DIR}/{input_model}')
        self.store('input_model', model)

        return COMPUTE_STATE


@app_state(COMPUTE_STATE)
class ComputeState(AppState):

    def register(self):
        self.register_transition(WRITE_STATE)

    def run(self):
        global_model = self.load('input_model')
        images = self.load('input_images')
        labels = self.load('input_labels')
        n_classes = self.load('n_classes')
        in_features = self.load('in_features')

        unclient_model = CNN(n_classes=n_classes, in_features=in_features)

        self.log('build train, val split for unlearning client')
        unclient_split = train_test_split(images, labels, stratify=labels, train_size=0.7)
        unclient_train_images, unclient_val_images, unclient_train_labels, unclient_val_labels = unclient_split

        unclient_train_set = CustomDataSet(images=images, labels=labels)

        unclient_train_loader = DataLoader(unclient_train_set, batch_size=256, shuffle=True, num_workers=2,
                                           persistent_workers=False)
        unclient_val_set = CustomDataSet(images=unclient_val_images, labels=unclient_val_labels)
        unclient_val_loader = DataLoader(unclient_val_set, batch_size=128, shuffle=True, num_workers=2,
                                         persistent_workers=False)

        unclient_train_loader = DataLoader(unclient_train_set, batch_size=256, shuffle=True, num_workers=2,
                                           persistent_workers=False)
        unclient_val_loader = DataLoader(unclient_val_set, batch_size=128, shuffle=True, num_workers=2,
                                         persistent_workers=False)

        self.log('initialization for unlearning')
        delta = self.load('delta')
        tau = self.load('tau')
        opt_lr = self.load('opt_lr')
        opt_weight_decay = self.load('opt_weight_decay')
        optimizer = optim.AdamW
        criterion = nn.CrossEntropyLoss()
        optimizer_params = {'lr': opt_lr, 'weight_decay': opt_weight_decay}
        untrain_optimizer = optim.AdamW(unclient_model.parameters(), lr=opt_lr, weight_decay=opt_weight_decay)

        self.log('starting federated unlearning')
        unfed_gym = FederatedUnlearnGym(unclient_model=unclient_model,
                                        unclient_train_loader=unclient_train_loader,
                                        unclient_val_loader=unclient_val_loader,
                                        model=global_model,
                                        client_train_loaders=None,
                                        val_loader=None,
                                        criterion=criterion,
                                        optimizer=optimizer, verbose=True,
                                        metric=metrics.balanced_accuracy_score,
                                        delta=delta, tau=tau)

        untrained_global_model, untrained_client_models, untrained_client_model = unfed_gym.untrain(
            client_untrain_epochs=5,
            federated_epochs=1,
            federated_rounds=1,
            untrain_optimizer=untrain_optimizer)

        return WRITE_STATE


@app_state(WRITE_STATE)
class WriteState(AppState):

    def register(self):
        self.register_transition(TERMINAL_STATE)

    def run(self):
        self.log('Output unlearned model')
        model = self.load('model')
        output_file = self.load('output_file')

        return TERMINAL_STATE




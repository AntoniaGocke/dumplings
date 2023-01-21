import bios
import numpy as np
import pandas as pd
from classes.CNN import CNN
from classes.UntrainFed import Gym, UnlearnGym, FederatedGym, FederatedUnlearnGym,ClientUnlearnGym
from FeatureCloud.app.engine.app import AppState, app_state, Role
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from sklearn import metrics

name = 'dumplings'

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

INITIAL_STATE = 'initial'
COMPUTE_STATE = 'compute'
AGGREGATE_STATE = 'aggregate'
WRITE_STATE = 'write'
TERMINAL_STATE = 'terminal'

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.


@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(
            COMPUTE_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log('Reading configuration file...')
        config = bios.read(f'{INPUT_DIR}/config.yaml')
        self.log(f'{config}')
        delta = config['delta']
        tau = config['tau']
        opt_lr = config['opt_lr']
        opt_weight_decay = config['opt_weight_decay']
        input_img = config['input_img']
        input_lab = config['input_lab']
        input_model = config['input_model']
        self.store('delta', config['delta'])
        self.store('tau', config['tau'])
        self.store('opt_lr', config['opt_lr'])
        self.store('opt_weight_decay', config['opt_weight_decay'])
        # needs to be changed to input from one client, not whole mnist
        self.store('input_img', config['input_img'])
        self.store('input_model', config['input_model'])
        self.log(f'Done reading configuration.')

        # read Images from npz
        self.log('Reading untraining images...')
        inputImgs = np.load(f'{INPUT_DIR}/{input_img}')
        self.store('input_images', inputImgs)

        #read labels from npz
        self.log('Reading untraining labels...')
        inputLabs = np.load(f'{INPUT_DIR}/{input_lab}')
        self.store('input_lab', inputLabs)

        # load model
        self.log('Reading model')
        model = CNN()
        model.load(f'{INPUT_DIR}/{input_model}')
        self.store('input_model', model)

        self.store('iteration', 0)

        return COMPUTE_STATE


@app_state(COMPUTE_STATE)
class ComputeState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE,
                                 role=Role.PARTICIPANT)
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(WRITE_STATE)

    def run(self):
        global_model = self.load('input_model')
        images = self.load('input_img')
        labels = self.load('input_lab')

        cl_model = CNN()

        self.log('build train, val split for unlearning client')
        unclient_split = train_test_split(images, labels, stratify=labels, train_size=0.7)
        unclient_train_images, unclient_val_images, unclient_train_labels, unclient_val_labels = unclient_split

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
                                        client_train_loaders=client_train_loaders,
                                        val_loader=val_loader,
                                        criterion=criterion,
                                        optimizer=optimizer, verbose=True,
                                        metric=metrics.balanced_accuracy_score,
                                        delta=delta, tau=tau, log=log)

        untrained_global_model, untrained_client_models, untrained_client_model = unfed_gym.untrain(
            client_untrain_epochs=5,
            federated_epochs=1,
            federated_rounds=1,
            untrain_optimizer=untrain_optimizer)

        if self.is_coordinator:
            return AGGREGATE_STATE
        else:
            return COMPUTE_STATE



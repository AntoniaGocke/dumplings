import bios
import numpy as np
import pandas as pd
from classes.CNN import CNN
from FeatureCloud.app.engine.app import AppState, app_state, Role

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
        optName = config['optName']
        opt_lr = config['opt_lr']
        opt_weight_decay = config['opt_weight_decay']
        input_file = config['input_file']
        input_model = config['input_model']
        self.store('iteration', 0)
        self.store('delta', config['delta'])
        self.store('tau', config['tau'])
        self.store('optName', config['optName'])
        self.store('opt_lr', config['opt_lr'])
        self.store('opt_weight_decay', config['opt_weight_decay'])
        self.store('input_file', config['input_file'])
        self.store('input_model', config['input_model'])
        self.log(f'Done reading configuration.')

        # read Images as npz
        self.log('Reading training data...')
        inputImgs = np.load(f'{INPUT_DIR}/{input_file}')
        self.store('input_files', inputImgs)

        # load model
        self.log('Reading model')
        model = CNN()
        model.load(f'{INPUT_DIR}/{input_model}')
        self.store('input_model', model)


        self.log('Preparing initial model...')
        lr = LinearRegression().fit(np.zeros(np.shape(df.drop(columns=target_column))), np.zeros(np.shape(df[target_column])))
        self.store('model', lr)

        self.store('iteration', 0)

        if self._coordinator:
            self.broadcast_data([lr._coef, lr._intercept, False])

        return COMPUTE_STATE


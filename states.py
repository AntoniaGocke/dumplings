import bios
import numpy as np
import pandas as pd

from FeatureCloud.app.engine.app import AppState, app_state, Role
from sklearn.linear_model import LinearRegression

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

@app_state(name= 'initial', role=Role.BOTH, app_name=name)
"""
read client model to unlearn
read global model to set constraint
"""
class Read(Initialization):



@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(
            COMPUTE_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log('Reading confirguration file...')
        config = bios.read(f'{INPUT_DIR}/config.yaml')
        self.log(f'{config}')
        max_iterations = config['max_iteration']
        input_file = config['data']
        input_sep = config['sep']
        target_column = config['target_value']
        self.store('iteration', 0)
        self.store('max_iteration', ['max_iter'])
        self.store('target_column', config['target_value'])
        self.store('input_file', config['data'])
        self.store('input_sep', config['sep'])
        self.log(f'Done reading configuration {max_iterations}.')

        self.log('Reading training data...')
        df = pd.read_csv(f'{INPUT_DIR}/{input_file}', input_sep)
        self.store('dataframe', df)

        self.log('Preparing initial model...')
        lr = LinearRegression().fit(np.zeros(np.shape(df.drop(columns=target_column))), np.zeros(np.shape(df[target_column])))
        self.store('model', lr)

        self.store('iteration', 0)

        if self._coordinator:
            self.broadcast_data([lr._coef, lr._intercept, False])

        return COMPUTE_STATE  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.


@app_state(COMPUTE_STATE)
class ComputeState(AppState):

    def register(self):
        self.register_transition(COMPUTE_STATE,
                                 role=Role.PARTICIPANT)  # We declare that 'terminal' state is accessible from the 'initial' state.
        self.register_transition(AGGREGATE_STATE, role=Role.COORDINATOR)
        self.register_transition(WRITE_STATE)

    def run(self):
        iteration = self.load('iteration')
        iteration += 1
        self.store('iteration', iteration)

        self.log(f'ITERATION {iteration}')

        # Receive global model from coordinator
        coef, intercept, done = self.await_data()

        if done:
            return WRITE_STATE

        model = self.load('model')
        model._coef = coef
        model._intercept = intercept

        self.log('Fitting model')
        df = self.load('dataframe')
        target_column = self.load('target_column')
        model.fit(df.drop(columns=target_column), df[target_column])
        self.store('model', model)

        self.log('Scoring model...')
        score = model.score(df.drop(columns=target_column), df[target_column])
        self.log(f'Score is {score}')

        self.send_data_to_coordinator(model.coef_, model.intercept_)

        if self.is_coordinator:
            return AGGREGATE_STATE  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.
        else:
            return COMPUTE_STATE


@app_state(AGGREGATE_STATE)
class AggregateState(AppState):

    def register(self):
        self.register_transition(
            COMPUTE_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log('Waiting for local models...')
        agg_coef, agg_intercept = self.aggregate_data()

        self.log('Aggregating models...')
        agg_coef = agg_coef / len(self.clients)
        agg_intercept = agg_intercept / len(self.clients)
        done = self.load('iteration') >= self.log(('max_iterations'))

        self.log('Broadcasting global model...')
        self.broadcast_data([agg_coef, agg_intercept])

        return COMPUTE_STATE  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.


@app_state(WRITE_STATE)
class WriteState(AppState):

    def register(self):
        self.register_transition(
            TERMINAL_STATE)  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        self.log('Predicting data...')
        df = self.load('dataframe')
        model = self.load('model')
        target_column = self.load('target_column')
        output_file = self.load('output_file')
        pred = model.predict(df.drop(cloumns=target_column))
        pd.DataFrame(data={'pred': pred}).tocsv(f'{OUTPUT_DIR}/{output_file}')

        return TERMINAL_STATE  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.

import joblib
import os
import pdb
import copy
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
from tqdm import tqdm

import pandas as pd
import numpy as np
from numpy import vstack

from trial_simulation.base import SequenceSimulationBase
from trial_simulation.data import SequencePatient, pad_batch_fn
from data.patient_data import SequencePatientBase
from trial_simulation.model import BuildModel, loss_function
from utils.helpers import (
    check_checkpoint_file, check_model_dir, check_model_config_file, make_dir_if_not_exist
)

class trial_data(Dataset):
    # load the dataset
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        #self.y = self.y.reshape((len(self.y), 1)))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X.iloc[idx, :].to_numpy(), self.y.iloc[idx, :].to_numpy()]

def prepare_data(X_train, y_train, batch_size =64):
    # load the dataset
    train = trial_data(X_train, y_train)
    # prepare data loaders. cannot shuffle during training
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=False)#, drop_last=True)
    return train_dl
    
class TWIN(SequenceSimulationBase):
    '''
    Implement a VAE based model for clinical trial patient digital twin simulation [1]_.
    
    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    order: list[str]
        The order of event types in each visits, e.g., ``['treatment', 'medication', 'adverse event']``.
        Visit = [treatment_events, medication_events, adverse_events], each event is a list of codes.

    freeze_event: str or list[str]
        The type(s) of event to be frozen during training and generation, e.g., ``['treatment']``.

    max_visit: int
        Maximum number of visits.

    emb_size: int
        Embedding size for encoding input event codes.
        
    latent_dim: int
        Size of final latent dimension between the encoder and decoder

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    batch_size: int
        Batch size when doing SGD optimization.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    num_worker: int
        Number of workers used to do dataloading during training.

    device: str
        Device to use for training, e.g., ``'cpu'`` or ``'cuda:0'``.

    experiment_id: str
        A unique identifier for the experiment.

    verbose: bool
        If True, print out training progress.

    k: int
        Number of neighbors for retrieval-augmented encoding.

    Notes
    -----
    .. [1] Trisha Das*, Zifeng Wang*, and Jimeng Sun. TWIN: Personalized Clinical Trial Digital Twin Generation. KDD'23.
    '''
    def __init__(self, 
        vocab_size, 
        order,
        freeze_event,
        max_visit=20,
        emb_size=64,
        hidden_dim=64,
        latent_dim=64,
        learning_rate = 1e-3,
        batch_size=200, 
        epochs=10, 
        num_worker=4,
        device='cpu',
        k=5,
        experiment_id = 'trial_simulation.sequence.twin',
        verbose=True):
        super().__init__(experiment_id)

        if isinstance(freeze_event, str):
            assert freeze_event in order, f'The specified freeze_event {freeze_event} is not in order {order}!'
            freeze_event = [freeze_event]

        if freeze_event is not None:
            for et in freeze_event:
                assert et in order, f'The specified freeze_event {et} is not in order {order}!'
        
        # build perturbing events
        if len(freeze_event) == 0 or freeze_event is None:
            perturb_event = copy.deepcopy(order)
        else:
            perturb_event = [et for et in order if et not in freeze_event]

        self.config = {
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'emb_size':emb_size,
            'hidden_dim': hidden_dim,
            'latent_dim':latent_dim,
            'device':device,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'epochs':epochs,
            'num_worker':num_worker,
            'orders':order,
            'output_dir': self.checkout_dir,
            'verbose': verbose,
            'freeze_event': freeze_event,
            'perturb_event': perturb_event,
            'k': k,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.device = device
        self._build_model()
    
    def fit(self, train_data):
        '''
        Fit the model with training data.

        Parameters
        ----------
        train_data: SequencePatientBase
            Training data.
        '''
        self._input_data_check(train_data)
        df_train_data = self._translate_sequence_to_df(train_data)

        # train the model for each event type
        for et in self.config['perturb_event']:
            model = self.models[et]
            model._fit_model(df_train_data)
        
        if self.config["verbose"]:
            print("Training finished.")

    def load_model(self, checkpoint):
        '''
        Load the learned model from the disk.

        Parameters
        ----------
        checkpoint: str
            - If a directory, the only checkpoint file `.model` will be loaded.
            - If a filepath, will load from this file;
            - If None, will load from `self.checkout_dir`.
        '''
        if checkpoint is None:
            checkpoint = self.checkout_dir

        checkpoint_filename = check_checkpoint_file(checkpoint, suffix='model')
        model = joblib.load(checkpoint_filename)
        self.__dict__.update(model.__dict__)

    def save_model(self, output_dir=None):
        '''Save the model to the given directory.

        Parameters
        ----------
        output_dir: str
            The directory to save the model. If None, then save to the default directory.
            `self.checkout_dir` is the default directory.
        '''
        if output_dir is None:
            output_dir = self.checkout_dir
        make_dir_if_not_exist(output_dir)
        ckpt_path = os.path.join(output_dir, 'twin.model')
        joblib.dump(self, ckpt_path)
        # save config
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)


    def predict(self, test_data, n_per_sample=None, n=None, verbose=False):
        '''
        Generate synthetic data for the given real data.
        
        Parameters
        ----------
        test_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.
                
        n_per_sample: int
            How many samples generated based on each indivudals.

        n: int
            How many samples in total will be generated. If None, then `n_per_sample` should be provided.
            Will be rounded to the closest multiple of `n_per_sample`.

        verbose: bool
            If True, print out generation progress.
        
        Returns
        -------
        fake_data: SequencePatient
            A `SequencePatient` contains generated patient records where 'v' corresponds to visit sequence of different events.
        '''
        self._input_data_check(test_data)

        if n is not None: assert isinstance(n, int), 'Input `n` should be integer.'
        if n_per_sample is not None: assert isinstance(n_per_sample, int), 'Input `n_per_sample` should be integer.'
        assert (not n_per_sample is None) or (not n is None), 'Either `n` or `n_per_sample` should be provided to generate.'
        n, n_per_sample = self._compute_n_per_sample(len(test_data), n, n_per_sample)

        # translate data
        df_data = self._translate_sequence_to_df(test_data)

        verbose = self.config['verbose'] or verbose
        if verbose:
            print(f"Generating {n} samples for {len(test_data)} individuals with {n_per_sample} samples per individual.")

        # generate data for each event type
        fake_data = {k:[] for k in self.config['orders']}
        for et in self.config['perturb_event']:
            model = self.models[et]
            pred = model.predict(df_data, n_per_sample=n_per_sample, verbose=verbose)
            fake_data[et] = pred
        
        # merge data
        fake_data = self._merge_generated_data(fake_data, df_data, n_per_sample)

        # build `SequencePatient`
        fake_data = self._translate_df_to_sequence(fake_data, test_data, n_per_sample=n_per_sample)
        return fake_data

    def _merge_generated_data(self, fake_data, df_data, n_per_sample):
        '''
        Merge generated data for each event type into a single `SequencePatient`.
        '''
        merged_data = []
        for et in self.config["orders"]:
            if et in self.config["perturb_event"]:
                # use the fake data
                syn = fake_data[et]
                syn = pd.concat(syn, axis=0).reset_index(drop=True)

            else: 
                # use the real data
                syn = df_data.iloc[:,df_data.columns.str.contains(et)].copy()
                if n_per_sample > 1:
                    syn = pd.DataFrame(np.repeat(syn.values, n_per_sample, axis=0), columns=syn.columns)            
            
            merged_data.append(syn)

        # build dataframe
        merged_data = pd.concat(merged_data, axis=1)

        # repeat the patient id and visit id for each sample
        syn_indexer = []
        for i in range(n_per_sample):
            syn_ind = df_data[["People","Visit"]].copy()
            syn_ind["People"] = syn_ind["People"].astype(str).apply(lambda x: "sample_{0}_twin_{1}".format(x, i))
            syn_indexer.append(syn_ind)
        syn_indexer = pd.concat(syn_indexer, axis=0).reset_index(drop=True)        

        # merge
        merged_data = pd.concat([syn_indexer, merged_data], axis=1)
        return merged_data
    
    def _remove_the_last_visit(self, data):
        data['Visit_']= data['Visit'].shift(-1)
        data.iloc[len(data)-1,-1]=-1
        data = data[data['Visit_']-data['Visit']==1]
        data = data.drop(columns =['Visit_'])
        return data

    def _translate_sequence_to_df(self, inputs):
        '''
        returns dataframe from SeqPatientBase
        '''
        inputs= inputs.visit
        column_names = ['People', 'Visit']
        for i in range(len(self.config['orders'])):
            for j in range(self.config['vocab_size'][i]):
              column_names.append(self.config['orders'][i]+'_'+str(j))

        visits = []
        for i in range(len(inputs)):#each patient
            if self.config['verbose'] and i % 100 == 0:
                print(f'Translating Data: Sample {i}/{len(inputs)}')

            for j in range(len(inputs[i])): #each visit
                binary_visit = [i, j]
                for k in range(len(self.config["orders"])): #orders indicate the order of events
                    event_binary= np.array([0]*self.config['vocab_size'][k])
                    event_binary[inputs[i][j][k]] = 1 #multihot from dense
                    binary_visit.extend(event_binary.tolist())

                visits.append(binary_visit)
        df = pd.DataFrame(visits, columns=column_names)
        return df


    def _translate_df_to_sequence(self, df, seqdata, n_per_sample=1):
        '''
        returns SeqPatientBase from df
        '''
        visits = []
        columns = []
        x_list = []

        def get_nnz(x):
            res = np.nonzero(x.to_list())[0].tolist()
            if len(res) == 0:
                res = [0]
            return res

        for k in self.config['orders']:
            columns.append(df.columns[df.columns.str.contains(k)].to_list())

        for idx, pid in enumerate(df.People.unique()):
            if self.config['verbose'] and idx % 100 == 0:
                print(f'Translating Data: Sample {idx}/{df.People.nunique()}')
            sample = []
            temp = df[df['People']==pid]
            for index, row in temp.iterrows():
                visit = []
                for cols in columns:
                    visit.append(get_nnz(row[cols]))
                sample.append(visit)
            visits.append(sample)
            x_list.append([pid, len(sample)])
        
        x_list = pd.DataFrame(x_list, columns=['pid', 'num_visits'])

        # copy metadata
        metadata = copy.deepcopy(seqdata.metadata)
        if getattr(seqdata, 'label', None) is not None:
            y = seqdata.label
            if n_per_sample > 1:
                y = np.tile(y, n_per_sample)
        else:
            y = None
        seqdata = SequencePatient(data={'v':visits, 'y': y, 'x': x_list}, metadata=metadata)
        return seqdata
    
    def _build_model(self):
        # build unimodal TWIN for the given event types
        self.models = {}
        for et in self.config['perturb_event']:
            self.models[et] = UnimodalTWIN(
                event_type=et,
                epochs=self.config['epochs'],
                vocab_size=self.config['vocab_size'],
                order=self.config['orders'],
                freeze_event=self.config['freeze_event'],
                max_visit=self.config['max_visit'],
                emb_size=self.config['emb_size'],
                hidden_dim=self.config['hidden_dim'],
                latent_dim=self.config['latent_dim'],
                device=self.config['device'],
                verbose=self.config['verbose'],
                k=self.config['k'],
                )

    def _input_data_check(self, inputs):
        from trial_simulation.data import SequencePatient
        assert isinstance(inputs, (SequencePatientBase, SequencePatient)), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase` or `SequencePatient`, find {type(inputs)} instead.'


class UnimodalTWIN(SequenceSimulationBase):
    '''
    Implement a VAE based model for clinical trial patient digital twin simulation [1]_.
    
    Parameters
    ----------
    vocab_size: list[int]
        A list of vocabulary size for different types of events, e.g., for diagnosis, procedure, medication.

    order: list[str]
        The order of event types in each visits, e.g., ``['treatment', 'medication', 'adverse event']``.
        Visit = [treatment_events, medication_events, adverse_events], each event is a list of codes.

    event_type: str or list[str]
        The type(s) of event to be modeled, e.g., ``'medication'`` or ``'adverse event'``.
        If a list is provided, then the model will be trained to model all event types in the list.

    freeze_event: str or list[str]
        The event type(s) that will be frozen during training, e.g., ``'medication'`` or ``'adverse event'``.

    max_visit: int
        Maximum number of visits.

    emb_size: int
        Embedding size for encoding input event codes.
        
    latent_dim: int
        Size of final latent dimension between the encoder and decoder

    learning_rate: float
        Learning rate for optimization based on SGD. Use torch.optim.Adam by default.

    batch_size: int
        Batch size when doing SGD optimization.

    epochs: int
        Maximum number of iterations taken for the solvers to converge.

    num_worker: int
        Number of workers used to do dataloading during training.

    device: str
        Device to use for training, e.g., ``'cpu'`` or ``'cuda:0'``.

    experiment_id: str
        A unique identifier for the experiment.

    verbose: bool
        If True, print out training progress.

    k: int
        Number of neighbors for retrieval-augmented encoding.

    Notes
    -----
    .. [1] Trisha Das*, Zifeng Wang*, and Jimeng Sun. TWIN: Personalized Clinical Trial Digital Twin Generation. KDD'23.
    '''
    def __init__(self,
        vocab_size,
        order,
        event_type= 'medication',
        freeze_event= None,
        max_visit=20,
        emb_size=64,
        hidden_dim=64,
        latent_dim=64,
        learning_rate=1e-3,
        batch_size=200,
        epochs=10,
        num_worker=4,
        device='cpu',
        experiment_id='trial_simulation.sequence.twin',
        verbose=True,
        k=5,
        ):
        super().__init__(experiment_id)

        assert isinstance(event_type, str), "UnimodalTWIN only supports one event type! Got {} instead.".format(event_type)

        if isinstance(freeze_event, str):
            freeze_event = [freeze_event]
        
        if freeze_event is not None:
            for et in freeze_event:
                assert et in order, "Event type {} not found in order {.".format(et, order)

        self.config = {
            'vocab_size':vocab_size,
            'max_visit':max_visit,
            'emb_size':emb_size,
            'hidden_dim': hidden_dim,
            'latent_dim':latent_dim,
            'device':device,
            'learning_rate':learning_rate,
            'batch_size':batch_size,
            'epochs':epochs,
            'num_worker':num_worker,
            'orders':order,
            'event_type': event_type,
            'freeze_event': freeze_event,
            'output_dir': self.checkout_dir,
            'verbose': verbose,
            'k': k,
            }
        self.config['total_vocab_size'] = sum(vocab_size)
        self.device = device
        self._build_model()
        
    def _build_model(self):
        self.model = BuildModel(
            hidden_dim = self.config['hidden_dim'],
            latent_dim=self.config['latent_dim'],
            vocab_size=self.config['vocab_size'],
            orders=self.config['orders'],
            event_type = self.config['event_type'],
            freeze_type=self.config['freeze_event'],
            device=self.device,
            epochs = self.config['epochs'],
            k = self.config['k']
            )
        self.model = self.model.to(self.device)

    def _next_step_df(self, data):
        # do not make in-place change
        data = data.copy()
        columns = pd.Series(data.columns)
        target_columns = columns[columns.str.startswith(self.config["event_type"])]
        if self.config["freeze_event"] is not None:
            freeze_type = self.config["freeze_event"]
            for t in freeze_type:
                add_target_columns = columns[columns.str.startswith(t)]
                target_columns = pd.concat([target_columns, add_target_columns])
        
        other_columns = columns[~columns.isin(target_columns)]
        other_columns = other_columns[~other_columns.isin(["Visit","People"])]

        def _create_new_col(x): 
            splits = x.split("_")
            num = int(splits[-1])
            return "###nxt###_{}_{}".format("_".join(splits[:-1]), num)

        nxt_target_columns = other_columns.apply(lambda x: _create_new_col(x))
        column_map = dict(zip(other_columns, nxt_target_columns))

        # create a new column by shifting up
        nxt_data = data[other_columns].shift(-1).rename(columns=column_map)
        data = pd.concat([data, nxt_data], axis=1)

        # remove NaN rows
        data = self._remove_the_last_visit(data)

        # build X and y
        y = data[nxt_target_columns]
        X = data[target_columns]
        return X, y

    def _train(self, train_dl, device, optimizer, vocab_size, batch_size, model, out_dir):
        if self.config["verbose"]:
            print("...Start training VAE...")
            print('--- event type: ', self.config['event_type'], '---')
            print('--- order: ', self.config['orders'], '---')
            print('--- freeze_event: ', self.config['freeze_event'], '---')
            print('--- vocab_size: ', vocab_size, '---')

        for epoch in range(self.config['epochs']):
            overall_loss = 0
            for batch_idx, (x, y) in enumerate(train_dl):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                n_cross_n = torch.matmul(x, x.T)
                top_5_index = torch.topk(n_cross_n, 4)
                ext_x=[]

                for i in range(x.shape[0]):
                    x_=[]
                    x_.append(x[i].tolist())
                    x_.extend(x[top_5_index.indices[i]].tolist())
                    ext_x.append(x_)

                ext_x= torch.as_tensor(ext_x)
                ext_x = ext_x.to(self.config["device"])
                x_hat, out_mean, log_var , out = model(ext_x)

                # x_hat should be the event_type reconstruction
                # y should be the other events to predict

                if self.config["freeze_event"] is not None:
                    x_indexes = model._create_non_freeze_indexes(x, model.freeze_dim_range)
                    x_tgt = x[:, x_indexes]
                else:
                    x_tgt = x

                loss = loss_function(x_tgt, x_hat, out_mean, log_var, out, y)
                overall_loss += loss.item()
                loss.backward()
                optimizer.step()

            if self.config["verbose"]:
                print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ".format(self.config["event_type"]), overall_loss / (batch_idx*batch_size))

        if self.config["verbose"]:
            print("Finish!!")

    def _fit_model(self, df, out_dir=None):
        X, y = self._next_step_df(df)
        train_dl= prepare_data(X, y, self.config['batch_size'])
        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self._train(train_dl, self.device, optimizer, self.config['vocab_size'], self.config['batch_size'], self.model, out_dir)

    def _input_data_check(self, inputs):
        from trial_simulation.data import SequencePatient
        assert isinstance(inputs, (SequencePatientBase, SequencePatient)), f'`trial_simulation.sequence` models require input training data in `SequencePatientBase` or `SequencePatient`, find {type(inputs)} instead.'

    def _remove_the_last_visit(self, data):
        data['Visit_']= data['Visit'].shift(-1)
        data.iloc[len(data)-1,-1]=-1
        data = data[data['Visit_']-data['Visit']==1]
        data = data.drop(columns =['Visit_'])
        return data

    def _translate_sequence_to_df(self, inputs):
        '''
        returns dataframe from SeqPatientBase
        '''
        inputs= inputs.visit
        column_names = ['People', 'Visit']
        for i in range(len(self.config['orders'])):
            for j in range(self.config['vocab_size'][i]):
              column_names.append(self.config['orders'][i]+'_'+str(j))

        visits = []
        for i in range(len(inputs)):#each patient
            if self.config['verbose'] and i % 100 == 0:
                print(f'Translating Data: Sample {i}/{len(inputs)}')

            for j in range(len(inputs[i])): #each visit
                binary_visit = [i, j]
                for k in range(len(self.config["orders"])): #orders indicate the order of events
                    event_binary= np.array([0]*self.config['vocab_size'][k])
                    event_binary[inputs[i][j][k]] = 1 #multihot from dense
                    binary_visit.extend(event_binary.tolist())

                visits.append(binary_visit)
        df = pd.DataFrame(visits, columns=column_names)
        return df

    def _generate_one_loop(self, X, y):
        '''
        generate one loop of the model
        '''
        dl = prepare_data(X, y, self.config['batch_size'])
        self.model.eval()
        x_hats, ins= list(), list()
        for i, (x, y) in enumerate(dl):
            # evaluate the model on the test set
            n_cross_n = torch.matmul(x, x.T)
    
            #print(x.shape)
            top_5_index = torch.topk(n_cross_n, 4)
    
            ext_x=[]
            for j in range(x.shape[0]):
                x_=[]
                x_.append(x[j].tolist())
                x_.extend(x[top_5_index.indices[j]].tolist())
                ext_x.append(x_)
    
            ext_x= torch.as_tensor(ext_x)
            ext_x = ext_x.to(self.config["device"])
    
            x_hat, mean, log_var, yhat= self.model(ext_x)
    
            inp = x.detach().cpu().numpy()
            x_hat = x_hat.detach().cpu().numpy()
            x_hat = x_hat.round()
            
            x_hats.append(x_hat)
            ins.append(inp)

        x_hats, ins =  vstack(x_hats), vstack(ins)
        return x_hats
    
    def predict(self, df_data, n_per_sample=None, verbose=False):
        # self._input_data_check(data)
        # df_data = self._translate_sequence_to_df(data)
        if n_per_sample is None:
            n_per_sample = 1
        
        x_hat_list = []
        for i in range(n_per_sample):
            if verbose:
                print(f'Generating loop {i+1}/{n_per_sample} for event type: `{self.config["event_type"]}`.')
            X, y = self._next_step_df(df_data)
            x_hats = self._generate_one_loop(X, y)

            # get the target event columns
            tgt_event = self.config["event_type"]
            tgt_columns = X.columns[X.columns.str.contains(tgt_event)].tolist()

            # fillnan with the original data for the last visits
            x_hats = pd.DataFrame(x_hats, columns=tgt_columns, index=X.index)
            left, x_hats = df_data[tgt_columns].align(x_hats, axis=0, join='left')
            x_hats = x_hats.fillna(left)            
            x_hat_list.append(x_hats)
            
        return x_hat_list

    def load_model(self, checkpoint):
        '''
        Load model and the pre-encoded trial embeddings from the given
        checkpoint dir.
        
        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.
            - If a directory, the only checkpoint file `*.pth.tar` will be loaded.
            - If a filepath, will load from this file.
        '''
        # checkpoint_filename = check_checkpoint_file(checkpoint)
        # config_filename = check_model_config_file(checkpoint)
        # state_dict = torch.load(checkpoint_filename, map_location=self.config['device'])
        # if config_filename is not None:
        #     config = self._load_config(config_filename)
        #     self.config = config
        # if self.config['event_type']=='medication':
        #     self.model.Encoder_med.load_state_dict(state_dict['encoder'])
        #     self.model.Decoder_med.load_state_dict(state_dict['decoder'])
        #     self.model.AE_pred.load_state_dict(state_dict['predictor'])
        # if self.config['event_type']=='adverse events':
        #     self.model.Encoder_ae.load_state_dict(state_dict['encoder'])
        #     self.model.Decoder_ae.load_state_dict(state_dict['decoder'])
        #     self.model.Med_pred.load_state_dict(state_dict['predictor'])
        raise NotImplementedError("UnimodalTWIN does not support `load_model`. Use `TWIN` instead.")

    # def _save_config(self, config, output_dir=None):
    #     temp_path = os.path.join(output_dir, self.config['event_type']+'_twin_config.json')
    #     with open(temp_path, 'w', encoding='utf-8') as f:
    #         f.write(
    #             json.dumps(config, indent=4)
    #         )

    def save_model(self, output_dir):
        '''
        Save the learned simulation model to the disk.
        Parameters
        ----------
        output_dir: str
            The dir to save the learned model.
        '''
        # make_dir_if_not_exist(output_dir)
        # self._save_config(config=self.config, output_dir=output_dir)
        # if (self.config['event_type']=='medication'):
        #     self._save_checkpoint({
        #             'encoder': self.model.Encoder_med.state_dict(),
        #             'decoder': self.model.Decoder_med.state_dict(),
        #             'predictor': self.model.AE_pred.state_dict()
        #         },output_dir=output_dir, filename='checkpoint_med.pth.tar')
        # if (self.config['event_type']=='adverse events'):
        #     self._save_checkpoint({
        #             'encoder': self.model.Encoder_ae.state_dict(),
        #             'decoder': self.model.Decoder_ae.state_dict(),
        #             'predictor': self.model.Med_pred.state_dict()
        #         },output_dir=output_dir, filename='checkpoint_ae.pth.tar')
        # print('Save the trained model to:', output_dir)
        raise NotImplementedError("UnimodalTWIN does not support `save_model`. Use `TWIN` instead.")

    def _create_non_freeze_indexes(self, x, freeze_dim_range):
        all_indexes = list(range(x.shape[-1]))
        for freeze_dim_range_ in freeze_dim_range:
            all_indexes = list(set(all_indexes) - set(range(freeze_dim_range_[0], freeze_dim_range_[1])))
        return all_indexes

    def _get_all_latent_vectors(self, data):
        """
        Generate memory bank of latent representations for all patients.
        This implements the indexed retriever mentioned in the paper.
        """
        self.model.eval()
        all_z = []
        loader = DataLoader(data, batch_size=self.config['batch_size'], shuffle=False, 
                          num_workers=self.config['num_worker'], collate_fn=pad_batch_fn)
        with torch.no_grad():
            for batch in loader:
                idx, x, y, orders, x_unformatted = batch
                x = x.to(self.device)
                
                # Extract the input for the current event type (without retrieval for memory bank generation)
                if self.model.freeze_dim > 0:
                    all_indexes = self.model._create_non_freeze_indexes(x, self.model.freeze_dim_range)
                    x_input = x[:,:, all_indexes].contiguous()
                else:
                    x_input = x
                
                # Use self-attention over patient's own visits (not retrieval)
                query = x_input[:, 0, :]
                keys = x_input[:, 1:, :]
                context, _ = self.model.Att(query, keys)

                # Generate latent representation
                out_mean, out_log_var = self.model.Encoder(context[:,0, :])
                z = self.model.reparameterization(out_mean, torch.exp(0.5 * out_log_var))
                all_z.append(z.cpu())
        
        self.model.train()
        return torch.cat(all_z, dim=0).to(self.device)

    def fit(self, train_data, outcome_model=None):
        self.outcome_model = outcome_model
        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])
        train_loader = DataLoader(dataset=train_data, 
                                batch_size=self.config['batch_size'], 
                                shuffle=True,
                                num_workers=self.config['num_worker'],
                                collate_fn=pad_batch_fn)

        for epoch in range(self.config['epochs']):
            # Generate memory bank for retrieval-augmented encoding
            memory_bank = self._get_all_latent_vectors(train_data)
            
            if self.config['verbose']:
                print(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            self.model.train()
            for i, batch in enumerate(tqdm(train_loader)):
                idx, x, y, orders, x_unformatted = batch
                x = x.to(self.device)
                
                # Forward pass with retrieval-augmented encoding
                x_hat, mean, log_var, pred_out = self.model(x, memory_bank=memory_bank, current_indices=idx)

                # Find which part of pred_out corresponds to which event type
                target_map = list(range(len(self.config['orders'])))
                target_map.pop(self.config['orders'].index(self.config['event_type']))
                
                if self.config['freeze_event'] is not None:
                    for fe in self.config['freeze_event']:
                        if fe in self.config['orders']:
                            target_map.pop(target_map.index(self.config['orders'].index(fe)))

                target_map_ordered = sorted(target_map)
                
                true_out = []
                x_unformatted_device = [t.to(self.device) if hasattr(t, 'to') else torch.tensor(t).to(self.device) for t in x_unformatted]

                for j, event_idx in enumerate(target_map_ordered):
                    true_out.append(x_unformatted_device[event_idx])
                
                true_out = torch.cat(true_out, dim=1)

                loss = loss_function(x = x_unformatted_device[self.config['orders'].index(self.config['event_type'])],
                                        x_hat = x_hat,
                                        mean = mean,
                                        log_var = log_var,
                                        AE_true= true_out,
                                        AE_out= pred_out)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0 and self.config['verbose']:
                    print(f"  Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
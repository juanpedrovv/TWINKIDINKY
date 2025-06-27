'''
Provide data functions for trial data simulation.
'''
import torch
import numpy as np
from pytrial.data.patient_data import TabularPatientBase, SequencePatientBase

def pad_batch_fn(batch):
    """
    Collate function for DataLoader to handle batches of sequences with different lengths.
    
    Parameters
    ----------
    batch : list
        List of tuples containing (index, x, y, order, x_unformatted)
    
    Returns
    -------
    tuple
        Collated batch with properly padded tensors
    """
    indices, sequences, labels, orders, x_unformatted = zip(*batch)
    
    # Convert to lists for easier handling
    indices = list(indices)
    labels = list(labels) if labels[0] is not None else None
    orders = orders[0] if orders[0] is not None else None
    
    # Handle sequences - convert to tensor format
    batch_tensors = []
    max_visits = max(len(seq) for seq in sequences)
    
    # Calculate total feature dimensions from orders
    if hasattr(sequences[0][0][0], '__len__'):
        # Multi-hot encoded format
        feature_dims = [len(sequences[0][0][i]) for i in range(len(sequences[0][0]))]
        total_features = sum(feature_dims)
    else:
        # Simple format
        total_features = len(sequences[0][0])
    
    for seq in sequences:
        # Pad sequence to max_visits
        padded_seq = []
        for visit in seq:
            if hasattr(visit[0], '__len__'):
                # Multi-hot format: flatten each visit
                flat_visit = []
                for event_type in visit:
                    if isinstance(event_type, (list, tuple)):
                        flat_visit.extend(event_type)
                    else:
                        flat_visit.append(event_type)
                padded_seq.append(flat_visit)
            else:
                padded_seq.append(visit)
        
        # Pad with zeros if needed
        while len(padded_seq) < max_visits:
            padded_seq.append([0] * total_features)
        
        batch_tensors.append(padded_seq)
    
    # Convert to tensor
    x_tensor = torch.FloatTensor(batch_tensors)
    
    # Handle x_unformatted - ensure it's in the right format
    x_unformatted_processed = []
    for x_unfmt in x_unformatted:
        if isinstance(x_unfmt, (list, tuple)):
            processed_unfmt = []
            for event_data in x_unfmt:
                if isinstance(event_data, (list, tuple, np.ndarray)):
                    processed_unfmt.append(torch.FloatTensor(event_data))
                else:
                    processed_unfmt.append(torch.FloatTensor([event_data]))
            x_unformatted_processed.append(processed_unfmt)
        else:
            x_unformatted_processed.append([torch.FloatTensor([x_unfmt])])
    
    return indices, x_tensor, labels, orders, x_unformatted_processed

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
    
    def __len__(self):
        return len(self.idx2word.keys())

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

class TabularPatient(TabularPatientBase):
    '''
    Tabular patient records.

    IGNORE:
    # MedGAN etc.: only support discrete events, should map continous values to quantiles then do simulation.
    # CopulaGaussian, CTGAN etc.: support both discrete and continuos values.
    IGNORE
    '''
    def __init__(self, df, metadata=None, transform=True):
        super().__init__(df, metadata=metadata, transform=transform)


class SequencePatient(SequencePatientBase):
    '''
    Load sequential patient inputs for longitudinal patient records generation.

    Parameters
    ----------
    data: dict
        A dict contains patient data in sequence and/or in tabular.
        Given dict:
            {
                'x': np.ndarray or pd.DataFrame
                    Static patient features in tabular form, typically those baseline information.

                'v': list or np.ndarray
                    Patient visit sequence in dense format or in tensor format (depends on the model input requirement.)
                    If in dense format, it is like [[c1,c2,c3],[c4,c5],...], with shape [n_patient, NA, NA];
                    If in tensor format, it is like [[0,1,1],[1,1,0],...] (multi-hot encoded), 
                        with shape [n_patient, max_num_visit, max_num_event].
                
                'y': np.ndarray or pd.Series
                    Target label for each patient if making risk detection, with shape [n_patient, n_class];
                    Target label for each visit if making some visit-level prediction with shape [n_patient, NA, n_class].
            }
    
    metadata: dict (optional)
        A dict contains configuration of input patient data.
        metadata:
            {
                'voc': dict[Voc]
                    Vocabulary contains the event index to the exact event name, has three keys in general:
                    'diag', 'med', 'prod', corresponding to diagnosis, medication, and procedure.
                    ``Voc`` object should have two functions: `idx2word` and `word2idx`.
                
                'visit': dict[str]
                    a dict contains the format of input data for processing input visit sequences.
                    `visit`: {
                        'mode': 'tensor' or 'dense',
                        'order': list[str] (required when `mode='tensor'`)
                    },

                'label': dict[str]
                    a dict contains the format of input data for processing input labels.
                    `label`: {
                        'mode': 'tensor' or 'dense',
                    }
                
                'max_visit': int
                    the maximum number of visits considered when building tensor inputs, ignored
                    when visit mode is dense.
                
                'n_num_feature': int
                    the number of numerical features in patients' baseline features
                
                'cat_cardinalities': list[int]
                    the cardinalities of each categorical features
            }

    '''
    visit = None
    feature = None
    label = None
    max_visit = None
    visit_voc_size = None
    visit_order = None

    metadata = {
        'voc': {},
        
        'visit':{
            'mode': 'dense',
            'order': ['diag', 'prod', 'med'],
            },

        'label':{
            'mode': 'tensor',
            },

        'max_visit': 20,
    }

    def __init__(self, data, metadata=None) -> None:
        super().__init__(data=data, metadata=metadata)
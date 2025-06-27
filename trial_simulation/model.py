import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder module that implements the embedding conversion from multi-hot vectors to dense embeddings
    as described in TWIN paper Equation (4): h^u_{n,t} = x^u_{n,t} · W^u_{emb}
    
    The encoder first converts multi-hot vectors to dense embeddings using a trainable embedding matrix,
    then encodes to latent space using variational autoencoder approach.
    """
    
    def __init__(self, vocab_size, event_order, freeze_order, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        input_dim = vocab_size[event_order]
        
        # W_emb: Trainable embedding matrix as described in TWIN paper Equation (4)
        # This converts multi-hot vector x^u_{n,t} ∈ {0,1}^l to dense embedding h^u_{n,t} ∈ R^d
        # W_emb^u ∈ R^{l×d} where l is vocab size and d is hidden_dim
        self.W_emb = nn.Linear(input_dim, hidden_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.W_emb.weight)
        
        # VAE encoder layers for latent space encoding
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        torch.nn.init.xavier_uniform_(self.FC_mean.weight)
        torch.nn.init.xavier_uniform_(self.FC_var.weight)
        
        self.LeakyReLU = nn.ReLU()
        self.training = True
        
    def forward(self, x):
        """
        Forward pass implementing TWIN paper Equation (4):
        h^u_{n,t} = x^u_{n,t} · W^u_{emb} ∈ R^d
        
        Args:
            x: Multi-hot encoded input vector x^u_{n,t} ∈ {0,1}^l
            
        Returns:
            mean: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Step 1: Convert multi-hot to dense embedding using trainable matrix W_emb
        # This implements Equation (4): h^u_{n,t} = x^u_{n,t} · W^u_{emb}
        h_emb = self.W_emb(x)  # Multi-hot to dense embedding conversion
        h_activated = self.LeakyReLU(h_emb)
        
        # Step 2: Encode to latent space (VAE encoding)
        mean     = self.FC_mean(h_activated)
        log_var  = self.FC_var(h_activated)
        return mean, log_var

class Decoder(nn.Module):
    """
    Decoder module that reconstructs multi-hot vectors from latent representations.
    """
    def __init__(self, latent_dim, hidden_dim, vocab_size, event_order):
        super(Decoder, self).__init__()
        output_dim = vocab_size[event_order]
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.FC_hidden.weight)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.FC_output.weight)
        
        self.LeakyReLU = nn.ReLU()
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    

class DotProductAttention(nn.Module):
    """
    Dot-product attention mechanism used in Retrieval Augmented Encoding.
    Computes attention weights and context vectors for combining retrieved neighbors.
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, value):
        batch_size, input_size = query.size(0),  value.size(1)
        
        score = torch.bmm(query.unsqueeze(1), value.transpose(1, 2))
        attn = F.softmax(score, 2)  
     
        context = torch.bmm(attn, value)

        return context, attn

def loss_function(x, x_hat, mean, log_var, AE_out, AE_true):
    """
    Combined loss function for TWIN model including:
    1. Reconstruction loss (binary cross entropy)
    2. KL divergence for VAE regularization
    3. Causality-preserving loss for temporal relationships
    """
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    ae_loss = nn.functional.binary_cross_entropy(AE_out, AE_true, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return (reproduction_loss + KLD) + ae_loss

class Predictor(nn.Module):
    """
    Causality Preserving Module (CPM) that predicts next timestep events
    based on current latent representations and treatments.
    """
    def __init__(self, latent_dim, hidden_dim, vocab_size, freeze_order, event_order):
        super(Predictor, self).__init__()
        freeze_dim = 0
        freeze_dim_range = [] # from where to where to freeze
        if freeze_order is not None:
            if isinstance(freeze_order, int):
                freeze_order = [freeze_order]
            for i in freeze_order:
                freeze_dim += vocab_size[i]
                start_idx = sum(vocab_size[:i])
                end_idx = sum(vocab_size[:i+1])
                freeze_dim_range.append([start_idx, end_idx])
        self.freeze_dim = freeze_dim
        self.freeze_dim_range = freeze_dim_range

        target_dim = sum(vocab_size) - vocab_size[event_order] - self.freeze_dim        
        self.FC_hidden = nn.Linear(latent_dim+freeze_dim, latent_dim+10)
        self.FC_hidden2 = nn.Linear(latent_dim +10, hidden_dim)
        self.FC_hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, target_dim)
        self.FC_output1 = nn.Linear(target_dim, target_dim)
        self.LeakyReLU= nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.bn1(self.LeakyReLU(self.FC_hidden2(h)))
        h     = self.LeakyReLU(self.FC_hidden3(h))
        h     = self.LeakyReLU(self.FC_output(h))
        next_time = torch.sigmoid(self.FC_output1(h))
        return next_time

class BuildModel(nn.Module):
    """
    Main TWIN model implementing:
    1. Multi-hot to dense embedding conversion (Equation 4)
    2. Retrieval Augmented Encoding (Equation 5) 
    3. Variational Autoencoder for generation
    4. Causality Preserving Module for temporal predictions
    """
    def __init__(self,
        hidden_dim,
        latent_dim,
        vocab_size,
        orders,
        event_type,
        freeze_type,
        device,
        epochs,
        k=5,
        ) -> None:
        super().__init__()
        self.event_type = event_type # either medication or adverse event
        self.device = device
        self.epochs = epochs
        self.k = k  # Number of neighbors for retrieval-augmented encoding

        # find event_tpye's index in orders
        target_order = orders.index(event_type)
      
        if not isinstance(vocab_size, list): vocab_size = [vocab_size]

        freeze_order = None
        if freeze_type is not None:
            if isinstance(freeze_type, str):
                freeze_type = [freeze_type]
            freeze_order = [orders.index(i) for i in freeze_type]

        self.freeze_order = freeze_order

        # Initialize encoder with explicit embedding matrix W_emb
        self.Encoder = Encoder(vocab_size=vocab_size, event_order=target_order, 
                             freeze_order=freeze_order, hidden_dim=hidden_dim, 
                             latent_dim=latent_dim)
        self.Decoder = Decoder(latent_dim, hidden_dim, vocab_size, target_order)

        # Causality Preserving Module (CPM) for temporal predictions
        self.Predictor = Predictor(latent_dim, hidden_dim, vocab_size, freeze_order, target_order)
        self.freeze_dim = self.Predictor.freeze_dim
        self.freeze_dim_range = self.Predictor.freeze_dim_range

        # Attention module for Retrieval Augmented Encoding
        self.Att = DotProductAttention()
        
        # Linear layer to combine original and augmented latent representations
        self.latent_combiner = nn.Linear(latent_dim * 2, latent_dim)


    def reparameterization(self, mean, var):
        """VAE reparameterization trick for sampling from latent distribution"""
        epsilon = torch.randn_like(var).to(self.device)        # sampling epsilon
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def forward(self, x, memory_bank=None, current_indices=None):
        """
        Forward pass implementing the full TWIN pipeline:
        
        1. Multi-hot to dense embedding conversion (Equation 4)
        2. Self-attention over patient's visit sequence  
        3. VAE encoding to latent space
        4. Retrieval Augmented Encoding (Equation 5) - if memory_bank provided
        5. Causality Preserving Module for temporal predictions
        6. Decoding back to multi-hot space
        
        Args:
            x: Input visit sequences [batch_size, max_visits, feature_dim]
            memory_bank: Latent representations of all patients for retrieval [N_patients, latent_dim]
            current_indices: Indices of current patients to avoid self-retrieval
            
        Returns:
            x_hat: Reconstructed multi-hot vectors
            out_mean: Mean of latent distribution  
            out_log_var: Log variance of latent distribution
            pred_out: Predictions for next timestep events
        """
        
        # Step 1: Extract input for current event type (excluding frozen events)
        if self.freeze_dim > 0:
            all_indexes = self._create_non_freeze_indexes(x, self.freeze_dim_range)
            x_input = x[:,:, all_indexes].contiguous()
        else:
            x_input = x

        # Step 2: Self-attention over patient's own visit sequence
        query = x_input[:, 0, :]  # Current visit
        keys = x_input[:, 1:, :]  # Previous visits
        context, _ = self.Att(query, keys)

        # Step 3: Encode to latent space using VAE with explicit embedding matrix W_emb
        # This implements Equation (4): h^u_{n,t} = x^u_{n,t} · W^u_{emb}
        out_mean, out_log_var = self.Encoder(context[:,0, :])
        z = self.reparameterization(out_mean, torch.exp(0.5 * out_log_var))
        
        # Step 4: Retrieval-Augmented Encoding (Equation 5) 
        # This implements the indexed retriever and attention-based combination
        z_augmented = z
        if memory_bank is not None and current_indices is not None:
            """
            Retrieval-Augmented Encoding Implementation:
            
            1. For each patient, find K most similar patients using dot-product similarity
            2. Retrieve their latent representations from memory bank
            3. Combine current patient with retrieved neighbors using attention
            4. This implements Equation (5): ĥ^u_{n,t} = Softmax(x_{n,t} · X_{n,K}^T) · H^u_{n,K}
            """
            batch_augmented = []
            
            for i, patient_idx in enumerate(current_indices):
                patient_z = z[i:i+1]  # Current patient's latent representation
                
                # Calculate dot-product similarity with all patients in memory bank
                # This implements the similarity calculation mentioned in the paper
                similarities = torch.matmul(patient_z, memory_bank.T)  # [1, num_patients]
                
                # Mask self-similarity to avoid retrieving the same patient
                if patient_idx < len(memory_bank):
                    similarities[0, patient_idx] = -1e9
                
                # Retrieve K most similar patients from the indexed retriever
                _, top_k_indices = torch.topk(similarities, min(self.k, memory_bank.size(0)), dim=1)
                neighbor_representations = memory_bank[top_k_indices.squeeze()]  # [K, latent_dim]
                
                # Combine current patient with retrieved neighbors
                # This prepares input for attention-based combination (Equation 5)
                combined_representations = torch.cat([patient_z, neighbor_representations], dim=0)  # [K+1, latent_dim]
                
                # Apply attention to get augmented representation
                # This implements the Softmax attention weighting in Equation (5)
                augmented_context, _ = self.Att(patient_z, combined_representations.unsqueeze(0))
                
                # Combine original and augmented representations
                patient_z_augmented = self.latent_combiner(
                    torch.cat([patient_z, augmented_context.squeeze(1)], dim=1)
                )
                
                batch_augmented.append(patient_z_augmented)
            
            z_augmented = torch.cat(batch_augmented, dim=0)

        # Step 5: Prepare input for Causality Preserving Module (CPM)
        # Combine augmented latent representation with frozen features (e.g., treatment)
        if self.freeze_dim > 0:
            z_predictor_input = torch.cat((z_augmented, x[:, 0, -self.freeze_dim:]), 1)
        else:
            z_predictor_input = z_augmented
            
        # Step 6: Generate predictions for next timestep using CPM
        pred_out = self.Predictor(z_predictor_input)
        
        # Step 7: Reconstruct multi-hot vectors (uses original z, not augmented)
        x_hat = self.Decoder(z)  
        
        return x_hat, out_mean, out_log_var, pred_out

    def _create_non_freeze_indexes(self, x, freeze_dim_range):
        """Helper function to create indexes for non-frozen event types"""
        all_indexes = list(range(x.shape[-1]))
        for freeze_dim_range_ in freeze_dim_range:
            all_indexes = list(set(all_indexes) - set(range(freeze_dim_range_[0], freeze_dim_range_[1])))
        return all_indexes
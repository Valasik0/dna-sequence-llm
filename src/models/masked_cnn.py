import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCNNGenerator(nn.Module):
    """
    CNN pro masked language modeling DNA sekvencí.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_filters: int = 128,
        filter_sizes: list = [3, 5, 7],
        dropout: float = 0.1
    ):
        super(MaskedCNNGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        
        # Embedding vrstva
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Více conv vrstev s různými kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in filter_sizes
        ])
        
        # Normalizace a dropout
        self.layer_norm = nn.LayerNorm(num_filters * len(filter_sizes))
        self.dropout = nn.Dropout(dropout)
        
        # Výstupní projekce
        self.output_projection = nn.Linear(num_filters * len(filter_sizes), vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - vstupní tokeny (některé mohou být <MASK>)
            attention_mask: [batch_size, seq_len] - maska pro padding (volitelné)
        Returns:
            logits: [batch_size, seq_len, vocab_size] - predikce pro každou pozici
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embedded = embedded.transpose(1, 2)   # [batch, embed_dim, seq_len]
        
        # Konvoluce s různými kernel sizes
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch, num_filters, seq_len]
            conv_outputs.append(conv_out)
        
        # Concatenace všech conv výstupů
        combined = torch.cat(conv_outputs, dim=1)  # [batch, num_filters*len(filter_sizes), seq_len]
        combined = combined.transpose(1, 2)        # [batch, seq_len, num_filters*len(filter_sizes)]
        
        # Normalizace a dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        # Projekce na vocabulary
        logits = self.output_projection(combined)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def predict_masked(self, input_ids, mask_token_id, device='cpu'):
        """
        Predikuje maskované tokeny v sekvenci.
        
        Args:
            input_ids: [seq_len] - sekvence s <MASK> tokeny
            mask_token_id: int - ID <MASK> tokenu
        Returns:
            predicted_ids: [seq_len] - sekvence s predikovanými tokeny
        """
        self.eval()
        input_batch = input_ids.unsqueeze(0).to(device)  # [1, seq_len]
        
        with torch.no_grad():
            logits = self.forward(input_batch)  # [1, seq_len, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # [1, seq_len]
            predictions = predictions.squeeze(0)  # [seq_len]
            
            # Nahraď pouze maskované pozice
            result = input_ids.clone()
            mask_positions = (input_ids == mask_token_id)
            result[mask_positions] = predictions[mask_positions]
            
            return result

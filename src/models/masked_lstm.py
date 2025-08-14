import torch
import torch.nn as nn
from typing import Optional
from .base import MaskedLanguageModel


class MaskedLSTMGenerator(MaskedLanguageModel):
    """
    Bidirectional LSTM pro masked language modeling DNA sekvencí.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__(vocab_size, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Embedding vrstva
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM pro masked modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,  # Bidirectional pro masked modeling!
            batch_first=True
        )
        
        # Výstupní projekce (hidden_dim * 2 kvůli bidirectional)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - vstupní tokeny s <MASK>
            attention_mask: [batch_size, seq_len] - maska pro padding
        Returns:
            logits: [batch_size, seq_len, vocab_size] - predikce pro každou pozici
        """
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embedded = self.dropout(embedded)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Projekce na vocabulary
        logits = self.output_projection(lstm_out)  # [batch, seq_len, vocab_size]
        
        return logits

    def get_init_config(self):
        """Konfig pro znovuvytvoření instance při načítání."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            # dropout lze odvodit z lstm/constructoru; ponecháme výchozí bezpečný
            'dropout': self.dropout.p if hasattr(self, 'dropout') else 0.0,
        }
    
    def predict_masked(self, input_ids, mask_token_id, device='cpu'):
        """
        Predikuje maskované tokeny v sekvenci.
        """
        self.eval()
        input_batch = input_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = self.forward(input_batch)
            predictions = torch.argmax(logits, dim=-1).squeeze(0)
            
            # Nahraď pouze maskované pozice
            result = input_ids.clone()
            mask_positions = (input_ids == mask_token_id)
            result[mask_positions] = predictions[mask_positions]
            
            return result

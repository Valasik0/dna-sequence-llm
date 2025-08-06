import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedTransformerGenerator(nn.Module):
    """
    Bidirectional Transformer pro masked language modeling.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super(MaskedTransformerGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer layers (bidirectional - bez causal masking!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - vstupní tokeny s <MASK>
            attention_mask: [batch_size, seq_len] - maska pro padding
        Returns:
            logits: [batch_size, seq_len, vocab_size] - predikce pro každou pozici
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [batch, seq_len, embed_dim]
        
        # Kombinace embeddings
        embeddings = token_emb + pos_emb
        embeddings = self.embedding_dropout(embeddings)
        
        # Attention mask pro padding (pokud je poskytnuta)
        transformer_mask = None
        if attention_mask is not None:
            # Transformer očekává bool masku (True = ignore)
            transformer_mask = ~attention_mask.bool()
        
        # Bidirectional transformer (bez causal masking!)
        transformer_out = self.transformer(embeddings, src_key_padding_mask=transformer_mask)
        
        # Projekce na vocabulary
        logits = self.output_projection(transformer_out)  # [batch, seq_len, vocab_size]
        
        return logits
    
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
    
    def predict_with_confidence(self, input_ids, mask_token_id, device='cpu'):
        """
        Predikuje maskované tokeny včetně confidence scores.
        """
        self.eval()
        input_batch = input_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = self.forward(input_batch)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1).squeeze(0)
            confidences = torch.max(probs, dim=-1)[0].squeeze(0)
            
            # Nahraď pouze maskované pozice
            result = input_ids.clone()
            mask_positions = (input_ids == mask_token_id)
            result[mask_positions] = predictions[mask_positions]
            
            return result, confidences

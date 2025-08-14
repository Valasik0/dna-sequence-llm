"""
DNA Sequence Models

Moduly pro maskované architektury neuronových sítí pro DNA sekvence.

Base třídy:
- BaseDNAModel: Základní třída pro všechny DNA modely
- MaskedLanguageModel: Základní třída pro masked language modely
- GenerativeLanguageModel: Základní třída pro generativní modely

Masked modely:
- MaskedCNNGenerator: CNN s bidirectional attention
- MaskedLSTMGenerator: LSTM s bidirectional processing  
- MaskedTransformerGenerator: Transformer encoder s bidirectional attention

Utilities:
- DNAModelTrainer: Univerzální trainer pro všechny architektury
- MaskedDNADataset: Dataset pro masked language modeling
"""

# Base třídy
from .base import BaseDNAModel, MaskedLanguageModel

# Maskované modely
from .masked_cnn import MaskedCNNGenerator
from .masked_lstm import MaskedLSTMGenerator
from .masked_transformer import MaskedTransformerGenerator

# Training utilities
from .trainer import DNAModelTrainer, MaskedDNADataset

__all__ = [
    # Base třídy
    'BaseDNAModel',
    'MaskedLanguageModel',
    
    # Modely
    'MaskedCNNGenerator',
    'MaskedLSTMGenerator', 
    'MaskedTransformerGenerator',
    
    # Training
    'DNAModelTrainer',
    'MaskedDNADataset'
]

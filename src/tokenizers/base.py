from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json


class DNATokenizer(ABC):
    """Abstraktní základní třída pro všechny DNA tokenizery."""
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.special_tokens = {
            'pad': '<PAD>',
            'unk': '<UNK>',
            'mask': '<MASK>'
        }
        
    @abstractmethod
    def build_vocab(self, sequences: List[str]) -> None:
        """Postaví slovník ze seznamu DNA sekvencí."""
        pass
    
    @abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        """Tokenizuje DNA sekvenci na seznam tokenů."""
        pass
    
    def encode(self, sequence: str) -> List[int]:
        """Převede DNA sekvenci na seznam ID tokenů."""
        tokens = self.tokenize(sequence)
        return [self.vocab.get(token, self.vocab.get(self.special_tokens['unk'], 0)) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Převede seznam ID tokenů zpět na DNA sekvenci."""
        tokens = [self.inverse_vocab.get(token_id, self.special_tokens['unk']) for token_id in token_ids]
        return ''.join(tokens).replace(self.special_tokens['pad'], '').replace(self.special_tokens['unk'], 'N')
    
    def batch_encode(self, sequences: List[str], max_length: Optional[int] = None, pad: bool = True) -> List[List[int]]:
        """Batch encoding s možností paddingu."""
        encoded = [self.encode(seq) for seq in sequences]
        
        if max_length is None:
            max_length = max(len(seq) for seq in encoded) if encoded else 0
            
        if pad:
            pad_id = self.vocab.get(self.special_tokens['pad'], 0)
            encoded = [seq[:max_length] + [pad_id] * max(0, max_length - len(seq)) for seq in encoded]
        
        return encoded
    
    def get_vocab_size(self) -> int:
        """Vrací velikost slovníku."""
        return self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Vrací slovník tokenů."""
        return self.vocab
    
    def __len__(self) -> int:
        return self.vocab_size

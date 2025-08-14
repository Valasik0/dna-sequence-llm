"""
DNA Tokenizer pro převod nukleotidových sekvencí na tokeny.
Extracted from original notebook and modularized.
"""

from typing import List, Dict, Optional
import torch
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for DNA tokenizers"""
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
    
    @abstractmethod
    def encode(self, sequence: str) -> List[int]:
        """Encode DNA sequence to tokens"""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to DNA sequence"""
        pass
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class DNATokenizer(BaseTokenizer):
    """
    Tokenizer pro DNA sekvence s různými metodami tokenizace.
    """
    
    def __init__(self, method: str = "single", k: int = 3):
        """
        Args:
            method: "single" (jednotlivé nukleotidy) nebo "kmer" (k-mer tokenizace)
            k: délka k-merů při kmer tokenizaci
        """
        super().__init__()
        self.method = method
        self.k = k
        
        if method == "single":
            self._build_single_vocab()
        elif method == "kmer":
            self._build_kmer_vocab(k)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _build_single_vocab(self):
        """Vocabulary pro jednotlivé nukleotidy + speciální tokeny"""
        tokens = ['A', 'T', 'G', 'C', '<MASK>']
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        # Speciální tokeny
        self.mask_token_id = self.vocab['<MASK>']
        self.pad_token_id = None  # V single mode nepoužíváme padding
    
    def _build_kmer_vocab(self, k: int):
        """Vocabulary pro k-mery"""
        nucleotides = ['A', 'T', 'G', 'C']
        
        # Generuj všechny možné k-mery
        def generate_kmers(length):
            if length == 1:
                return nucleotides
            else:
                shorter_kmers = generate_kmers(length - 1)
                return [kmer + nuc for kmer in shorter_kmers for nuc in nucleotides]
        
        kmers = generate_kmers(k)
        special_tokens = ['<MASK>', '<PAD>', '<UNK>']
        
        all_tokens = special_tokens + kmers
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        # Speciální tokeny
        self.mask_token_id = self.vocab['<MASK>']
        self.pad_token_id = self.vocab['<PAD>']
        self.unk_token_id = self.vocab['<UNK>']
    
    def encode(self, sequence: str) -> List[int]:
        """Encode DNA sequence to token IDs"""
        sequence = sequence.upper().replace('N', 'A')  # Replace unknown with A
        
        if self.method == "single":
            return [self.vocab.get(char, self.vocab['<MASK>']) for char in sequence]
        
        elif self.method == "kmer":
            tokens = []
            for i in range(len(sequence) - self.k + 1):
                kmer = sequence[i:i + self.k]
                token_id = self.vocab.get(kmer, self.unk_token_id)
                tokens.append(token_id)
            return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to DNA sequence"""
        if self.method == "single":
            return ''.join([self.reverse_vocab.get(token, 'N') for token in tokens])
        
        elif self.method == "kmer":
            # Pro k-mery rekonstruujeme sekvenci
            if not tokens:
                return ""
            
            # První k-mer
            sequence = self.reverse_vocab.get(tokens[0], 'N' * self.k)
            
            # Překrývající se k-mery
            for token in tokens[1:]:
                kmer = self.reverse_vocab.get(token, 'N' * self.k)
                sequence += kmer[-1]  # Přidej pouze poslední nukleotid
            
            return sequence
    
    def batch_encode(self, sequences: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """Encode batch of sequences to tensor"""
        encoded = [self.encode(seq) for seq in sequences]
        
        if max_length:
            # Truncate or pad to max_length
            for i, seq in enumerate(encoded):
                if len(seq) > max_length:
                    encoded[i] = seq[:max_length]
                elif len(seq) < max_length and self.pad_token_id is not None:
                    encoded[i] = seq + [self.pad_token_id] * (max_length - len(seq))
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs"""
        special_tokens = {}
        if hasattr(self, 'mask_token_id'):
            special_tokens['mask'] = self.mask_token_id
        if hasattr(self, 'pad_token_id') and self.pad_token_id is not None:
            special_tokens['pad'] = self.pad_token_id
        if hasattr(self, 'unk_token_id'):
            special_tokens['unk'] = self.unk_token_id
        return special_tokens

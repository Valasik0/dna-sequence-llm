from typing import List
from .base import DNATokenizer


class NucleotideTokenizer(DNATokenizer):
    """Tokenizer pro maskovaný přístup (GROVER-style) - pouze A,C,G,T + MASK."""
    
    def __init__(self):
        super().__init__()
        self.nucleotides = {'A', 'C', 'G', 'T'}
        
    def build_vocab(self, sequences: List[str]) -> None:
        """Postaví slovník pro maskovaný přístup."""
        vocab_set = set()
        
        for token in self.special_tokens.values():
            vocab_set.add(token)
        
        # Základní nukleotidy
        vocab_set.update(self.nucleotides)
        
        # Vytvoř slovník
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Masked tokenizer: {self.vocab_size} tokens")
        
    def mask_sequence(self, tokens: List[str], mask_prob: float = 0.15) -> tuple:
        """
        Maskuje náhodné tokeny v sekvenci.
        
        Args:
            tokens: Seznam nukleotidů
            mask_prob: Pravděpodobnost maskování každého tokenu
            
        Returns:
            (masked_tokens, target_tokens): Tuple s maskovanými a originálními tokeny
        """
        import random
        
        masked_tokens = tokens.copy()
        target_tokens = [-100] * len(tokens)  # -100 = ignore v CrossEntropyLoss
        
        for i, token in enumerate(tokens):
            if random.random() < mask_prob:
                # Maskuj tento token
                masked_tokens[i] = self.special_tokens['mask']
                target_tokens[i] = self.vocab.get(token, self.vocab[self.special_tokens['unk']])
        
        return masked_tokens, target_tokens
    
    def tokenize(self, sequence: str) -> List[str]:
        """Tokenizuje sekvenci na jednotlivé nukleotidy."""
        return [char for char in sequence.upper() if char in self.nucleotides]

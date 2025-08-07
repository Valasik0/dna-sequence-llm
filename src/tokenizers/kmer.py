from typing import List
from collections import Counter
from .base import DNATokenizer


class KmerTokenizer(DNATokenizer):
    """Tokenizer který tokenizuje DNA sekvence na k-mery (pouze A,C,G,T)."""
    
    def __init__(self, k: int = 3, min_frequency: int = 1, overlap: bool = True):
        super().__init__()
        self.k = k
        self.min_frequency = min_frequency
        self.overlap = overlap
        self.nucleotides = {'A', 'C', 'G', 'T'}
        
    def _extract_kmers(self, sequence: str) -> List[str]:
        """Extrahuje k-mery ze sekvence (pouze A,C,G,T)."""
        # Filtruj sekvenci na pouze validní nukleotidy
        filtered_sequence = ''.join(char for char in sequence.upper() if char in self.nucleotides)
        
        if len(filtered_sequence) < self.k:
            return []
        
        kmers = []
        step = 1 if self.overlap else self.k
        
        for i in range(0, len(filtered_sequence) - self.k + 1, step):
            kmer = filtered_sequence[i:i + self.k]
            # Ještě jednou ověř, že k-mer obsahuje pouze validní nukleotidy
            if all(char in self.nucleotides for char in kmer):
                kmers.append(kmer)
        
        return kmers
        
    def build_vocab(self, sequences: List[str]) -> None:
        """Postaví slovník k-merů na základě trénovacích sekvencí."""
        all_kmers = []
        
        # Extrahuj všechny k-mery
        for sequence in sequences:
            kmers = self._extract_kmers(sequence)
            all_kmers.extend(kmers)
        
        # Spočítej frekvence
        kmer_counts = Counter(all_kmers)
        
        # Filtruj k-mery podle min_frequency
        frequent_kmers = {kmer for kmer, count in kmer_counts.items() 
                         if count >= self.min_frequency}
        
        # Vytvoř slovník
        vocab_set = set()
        
        # Přidej speciální tokeny
        for token in self.special_tokens.values():
            vocab_set.add(token)
        
        # Přidej časté k-mery
        vocab_set.update(frequent_kmers)
        
        # Vytvoř slovník
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"K-mer slovník vytvořen: {len(frequent_kmers)} k-merů (k={self.k}, "
              f"min_freq={self.min_frequency}, overlap={self.overlap})")
    
    def tokenize(self, sequence: str) -> List[str]:
        """Tokenizuje sekvenci na k-mery (pouze A,C,G,T)."""
        return self._extract_kmers(sequence)
    
    def mask_sequence(self, tokens: List[str], mask_prob: float = 0.15) -> tuple:
        """
        Maskuje náhodné k-mery v sekvenci.
        
        Args:
            tokens: Seznam k-merů
            mask_prob: Pravděpodobnost maskování každého k-meru
            
        Returns:
            (masked_tokens, target_tokens): Tuple s maskovanými a originálními tokeny
        """
        import random
        
        masked_tokens = tokens.copy()
        target_tokens = [-100] * len(tokens)  # -100 = ignore v CrossEntropyLoss
        
        for i, token in enumerate(tokens):
            if random.random() < mask_prob:
                # Maskuj tento k-mer
                masked_tokens[i] = self.special_tokens['mask']
                target_tokens[i] = self.vocab.get(token, self.vocab[self.special_tokens['unk']])
        
        return masked_tokens, target_tokens

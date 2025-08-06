"""
Base classes for DNA sequence models and training.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseDNAModel(nn.Module, ABC):
    """
    Abstraktní základní třída pro všechny DNA modely.
    
    Definuje společné rozhraní pro generativní i masked modely.
    """
    
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_type = self.__class__.__name__
        
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass modelu.
        
        Args:
            input_ids: [batch_size, seq_len] - vstupní tokeny
            attention_mask: [batch_size, seq_len] - maska pro padding
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] - výstupní logits
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Vrací informace o modelu."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'vocab_size': self.vocab_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def count_parameters(self) -> int:
        """Spočítá celkový počet parametrů."""
        return sum(p.numel() for p in self.parameters())


class MaskedLanguageModel(BaseDNAModel):
    """
    Společná base třída pro masked language modely (GROVER/BERT-style).
    """
    
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.mask_token_id = kwargs.get('mask_token_id', 2)  # Default <MASK> token ID
    
    def predict_masked_tokens(self, input_ids: torch.Tensor, mask_token_id: int, 
                             device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predikuje maskované tokeny v sekvenci.
        
        Args:
            input_ids: [seq_len] - vstupní sekvence s <MASK> tokeny
            mask_token_id: ID <MASK> tokenu
            device: zařízení pro výpočet
            
        Returns:
            predictions: predikované tokeny
            confidences: confidence scores
        """
        self.eval()
        input_batch = input_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = self.forward(input_batch)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1).squeeze(0)
            confidences = torch.max(probs, dim=-1)[0].squeeze(0)
            
            # Nahraď pouze maskované pozice
            result = input_ids.clone()
            mask_positions = (input_ids == mask_token_id)
            result[mask_positions] = predictions[mask_positions]
            
            return result, confidences
    
    def compute_masked_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Spočítá accuracy pouze na maskovaných pozicích."""
        mask_positions = (labels != -100)
        if mask_positions.sum() == 0:
            return 0.0
            
        predictions = torch.argmax(logits.view(-1, logits.size(-1)), dim=-1)
        correct = (predictions == labels.view(-1)) & mask_positions.view(-1)
        
        return correct.sum().item() / mask_positions.sum().item()
    
    def save_model(self, save_path: str, tokenizer=None, training_results: Dict[str, Any] = None):
        """
        Uloží model s tokenizerem a training výsledky.
        
        Args:
            save_path: cesta pro uložení modelu
            tokenizer: tokenizer pro uložení
            training_results: výsledky tréninku
        """
        # Přidej specifické informace pro masked model
        if training_results is None:
            training_results = {}
        
        training_results['model_specific'] = {
            'mask_token_id': self.mask_token_id,
            'model_style': 'masked_language_model'
        }
        
        save_data = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'model_type': self.model_type,
                'vocab_size': self.vocab_size,
                'model_class': self.__class__.__name__
            },
            'model_info': self.get_model_info()
        }
        
        # Přidej tokenizer pokud je dostupný
        if tokenizer is not None:
            save_data['tokenizer'] = {
                'vocab': getattr(tokenizer, 'vocab', None),
                'inverse_vocab': getattr(tokenizer, 'inverse_vocab', None),
                'tokenizer_class': tokenizer.__class__.__name__
            }
        
        # Přidej training výsledky
        if training_results is not None:
            save_data['training_results'] = training_results
        
        torch.save(save_data, save_path)
        print(f"💾 Model uložen: {save_path}")


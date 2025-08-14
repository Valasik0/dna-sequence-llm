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

    # --- Konfigurační metadata pro uložení/načtení ---
    def get_init_config(self) -> Dict[str, Any]:
        """
        Vrátí minimální konfig pro znovuvytvoření modelu při načítání.
        Subtřídy by měly přetížit a vrátit své hyperparametry použité v __init__.
        """
        return {
            'vocab_size': self.vocab_size
        }

    @staticmethod
    def _resolve_model_class(class_name: str):
        """Vrátí třídu modelu podle jména uloženého v checkpointu."""
        try:
            # Importy až při běhu, abychom předešli cyklickým importům
            from .masked_lstm import MaskedLSTMGenerator
            from .masked_cnn import MaskedCNNGenerator
            from .masked_transformer import MaskedTransformerGenerator
        except Exception:
            MaskedLSTMGenerator = MaskedCNNGenerator = MaskedTransformerGenerator = None  # type: ignore
        mapping = {
            'MaskedLSTMGenerator': MaskedLSTMGenerator,
            'MaskedCNNGenerator': MaskedCNNGenerator,
            'MaskedTransformerGenerator': MaskedTransformerGenerator,
        }
        model_cls = mapping.get(class_name)
        if model_cls is None:
            raise ValueError(f"Neznámá třída modelu v checkpointu: {class_name}")
        return model_cls

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None):
        """
        Načte model (a případně tokenizer) přímo z .pt souboru bez nutnosti ruční
        rekonstrukce architektury. Vrací tuple (model, tokenizer | None, training_results | None).
        
        Pozn.: Je to classmethod kvůli ergonomii: BaseDNAModel.load_model(...)
        nebo MaskedLSTMGenerator.load_model(...). Skutečná třída se vezme z checkpointu.
        """
        map_location = 'cpu' if (device is None or device == 'cpu') else device
        checkpoint = torch.load(path, map_location=map_location)

        model_cfg = checkpoint.get('model_config', {})
        class_name = model_cfg.get('model_class') or model_cfg.get('model_type')
        if not class_name:
            raise ValueError("Checkpoint neobsahuje 'model_class' ani 'model_type'.")

        # Získej třídu modelu a init config
        model_cls = cls._resolve_model_class(class_name)
        init_cfg = model_cfg.get('init_config') or {}

        # Fallback: odvoz hyperparametrů z tvarů vah (pro LSTM)
        state = checkpoint.get('model_state_dict', {})
        if not init_cfg:
            # Minimálně vocab_size
            # Zkus odhadnout z embeddingu/Projection
            vocab_size = None
            if 'embedding.weight' in state:
                vocab_size = state['embedding.weight'].shape[0]
            elif 'output_projection.weight' in state:
                vocab_size = state['output_projection.weight'].shape[0]
            if vocab_size is None:
                raise ValueError("Nelze odvodit vocab_size z checkpointu. Uložte znovu s init_config.")

            init_cfg['vocab_size'] = vocab_size

            # Specifika pro MaskedLSTMGenerator (bezpečný odhad)
            if class_name == 'MaskedLSTMGenerator':
                emb_dim = None
                hid_dim = None
                num_layers = 1
                if 'embedding.weight' in state:
                    emb_dim = state['embedding.weight'].shape[1]
                # LSTM weight_ih_l0: (4*hidden_dim, embedding_dim)
                # LSTM weight_hh_l0: (4*hidden_dim, hidden_dim)
                if 'lstm.weight_hh_l0' in state:
                    hid_dim = state['lstm.weight_hh_l0'].shape[1]
                # Spočítej vrstvy (bez _reverse)
                layer_keys = [k for k in state.keys() if k.startswith('lstm.weight_ih_l') and not k.endswith('_reverse')]
                if layer_keys:
                    num_layers = len(layer_keys)

                if emb_dim is not None:
                    init_cfg['embedding_dim'] = emb_dim
                if hid_dim is not None:
                    init_cfg['hidden_dim'] = hid_dim
                init_cfg['num_layers'] = num_layers
                # dropout neovlivní tvar vah, bezpečně nastavíme 0.0/0.2
                init_cfg.setdefault('dropout', 0.0 if num_layers <= 1 else 0.2)

        # Vytvoř instanci a nahraj váhy
        model = model_cls(**init_cfg)
        model.load_state_dict(state)
        if device and device != 'cpu':
            model.to(device)

        # Rekonstrukce tokenizeru (pokud je v checkpointu)
        tokenizer = None
        tok_data = checkpoint.get('tokenizer')
        if tok_data:
            try:
                from tokenizers import KmerTokenizer, NucleotideTokenizer
                tok_class = (tok_data.get('tokenizer_class') or '').lower()
                if 'kmer' in tok_class:
                    k = tok_data.get('k') or tok_data.get('config', {}).get('k', 3)
                    tokenizer = KmerTokenizer(k=k)
                    tokenizer.build_vocab([])
                    inner = getattr(tokenizer, '_tokenizer', None)
                    if inner is not None:
                        inner.vocab = tok_data.get('vocab', {})
                        inv = tok_data.get('inverse_vocab')
                        if inv:
                            inner.reverse_vocab = {int(k): v for k, v in inv.items()}
                        else:
                            inner.reverse_vocab = {v: k for k, v in inner.vocab.items()}
                        specs = tok_data.get('special_tokens', {})
                        inner.mask_token_id = inner.vocab.get(specs.get('mask', '<MASK>'))
                        inner.pad_token_id = inner.vocab.get(specs.get('pad', '<PAD>'))
                        inner.unk_token_id = inner.vocab.get(specs.get('unk', '<UNK>'))
                else:
                    tokenizer = NucleotideTokenizer()
                    tokenizer.build_vocab([])
                    inner = getattr(tokenizer, '_tokenizer', None)
                    if inner is not None:
                        inner.vocab = tok_data.get('vocab', {})
                        inner.reverse_vocab = {v: k for k, v in inner.vocab.items()}
                        specs = tok_data.get('special_tokens', {})
                        inner.mask_token_id = inner.vocab.get(specs.get('mask', '<MASK>'))
            except Exception:
                tokenizer = None  # fallback: tokenizer nelze rekonstruovat

        training_results = checkpoint.get('training_results')
        return model, tokenizer, training_results


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
                'model_class': self.__class__.__name__,
                'init_config': self.get_init_config()
            },
            'model_info': self.get_model_info()
        }
        
        # Přidej tokenizer pokud je dostupný
        if tokenizer is not None:
            # Podpora jak wrapperů (KmerTokenizer/NucleotideTokenizer), tak DNATokenizer
            tok_payload: Dict[str, Any] = {
                'tokenizer_class': tokenizer.__class__.__name__
            }
            # Wrappery mohou mít vnitřní _tokenizer
            inner = getattr(tokenizer, '_tokenizer', None)
            if inner is not None:
                tok_payload.update({
                    'vocab': getattr(inner, 'vocab', None),
                    'inverse_vocab': getattr(inner, 'reverse_vocab', None),
                })
                # Zachyť k, special tokens a další
                for attr in ('k', 'min_frequency', 'overlap'):
                    if hasattr(tokenizer, attr):
                        tok_payload[attr] = getattr(tokenizer, attr)
            else:
                tok_payload.update({
                    'vocab': getattr(tokenizer, 'vocab', None),
                    'inverse_vocab': getattr(tokenizer, 'inverse_vocab', None),
                })
                for attr in ('k', 'min_frequency', 'overlap'):
                    if hasattr(tokenizer, attr):
                        tok_payload[attr] = getattr(tokenizer, attr)
            # Special tokens pokud existují
            specs = getattr(tokenizer, 'special_tokens', None)
            if specs is not None:
                tok_payload['special_tokens'] = specs

            save_data['tokenizer'] = tok_payload
        
        # Přidej training výsledky
        if training_results is not None:
            save_data['training_results'] = training_results
        
        torch.save(save_data, save_path)
        print(f"💾 Model uložen: {save_path}")


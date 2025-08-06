"""
Universal trainer for DNA sequence models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json
import random
from .base import BaseDNAModel, MaskedLanguageModel


class MaskedDNADataset(Dataset):
    """Dataset pro masked language modeling."""
    
    def __init__(self, sequence: str, tokenizer, seq_length: int = 256, 
                 mask_prob: float = 0.15, overlap_ratio: float = 0.5):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask_prob = mask_prob
        
        # Rozdělení sekvence na chunky s překryvem
        step = int(seq_length * (1 - overlap_ratio))
        self.sequences = []
        
        for i in range(0, len(sequence) - seq_length, step):
            chunk = sequence[i:i + seq_length]
            if len(chunk) == seq_length:
                self.sequences.append(chunk)
        
        print(f"📦 Dataset: {len(self.sequences)} sekvencí délky {seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenizace
        tokens = self.tokenizer.tokenize(sequence)
        
        # Masking (pokud tokenizer podporuje)
        if hasattr(self.tokenizer, 'mask_sequence'):
            masked_tokens, targets = self.tokenizer.mask_sequence(tokens, self.mask_prob)
        else:
            # Fallback pro tokenizery bez mask_sequence
            masked_tokens = tokens
            targets = [-100] * len(tokens)
        
        # Encode to IDs
        input_ids = [self.tokenizer.vocab.get(token, self.tokenizer.vocab.get('<UNK>', 1)) 
                    for token in masked_tokens]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(targets, dtype=torch.long)
        }


class DNAModelTrainer:
    """
    Univerzální trainer pro DNA modely.
    
    Podporuje jak masked language modeling tak generativní modely.
    """
    
    def __init__(self, model: BaseDNAModel, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Training historie
        self.train_history = []
        self.val_history = []
        
        print(f"🔧 Trainer inicializován:")
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Parametry: {model.count_parameters():,}")
        print(f"   Device: {self.device}")
    
    def create_dataset(self, sequence: str, seq_length: int = 256, 
                      mask_prob: float = 0.15, overlap_ratio: float = 0.5) -> MaskedDNADataset:
        """Vytvoří dataset ze sekvence."""
        return MaskedDNADataset(sequence, self.tokenizer, seq_length, mask_prob, overlap_ratio)
    
    def create_dataloaders(self, dataset: Dataset, train_ratio: float = 0.8, 
                          batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Vytvoří train a validation DataLoadery."""
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"📊 Data split: Train={len(train_dataset)}, Val={len(val_dataset)}")
        return train_loader, val_loader
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch_num: int) -> Tuple[float, float]:
        """Jeden epoch tréninku."""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Accuracy (pro masked modely)
            if isinstance(self.model, MaskedLanguageModel):
                accuracy = self.model.compute_masked_accuracy(logits, labels)
            else:
                # Pro generativní modely - accuracy na všech pozicích
                predictions = torch.argmax(logits.view(-1, logits.size(-1)), dim=-1)
                valid_positions = (labels.view(-1) != -100)
                if valid_positions.sum() > 0:
                    accuracy = ((predictions == labels.view(-1)) & valid_positions).float().mean().item()
                else:
                    accuracy = 0.0
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.3f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validace modelu."""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Accuracy
                if isinstance(self.model, MaskedLanguageModel):
                    accuracy = self.model.compute_masked_accuracy(logits, labels)
                else:
                    predictions = torch.argmax(logits.view(-1, logits.size(-1)), dim=-1)
                    valid_positions = (labels.view(-1) != -100)
                    if valid_positions.sum() > 0:
                        accuracy = ((predictions == labels.view(-1)) & valid_positions).float().mean().item()
                    else:
                        accuracy = 0.0
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train(self, sequence: str, num_epochs: int = 5, learning_rate: float = 1e-3,
              batch_size: int = 32, seq_length: int = 256, mask_prob: float = 0.15,
              train_ratio: float = 0.8, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Kompletní training pipeline.
        
        Args:
            sequence: DNA sekvence pro trénink
            num_epochs: počet epoch
            learning_rate: learning rate
            batch_size: velikost batche
            seq_length: délka sekvencí
            mask_prob: pravděpodobnost maskování
            train_ratio: poměr train/val split
            save_path: cesta pro uložení modelu
            
        Returns:
            training_results: výsledky tréninku
        """
        
        dataset = self.create_dataset(sequence, seq_length, mask_prob)
        train_loader, val_loader = self.create_dataloaders(dataset, train_ratio, batch_size)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch + 1)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Uložit historii
            self.train_history.append({'loss': train_loss, 'accuracy': train_acc})
            self.val_history.append({'loss': val_loss, 'accuracy': val_acc})
            
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.3f}")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.3f}")
        
        # Výsledky
        results = {
            'model_info': self.model.get_model_info(),
            'training_config': {
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'seq_length': seq_length,
                'mask_prob': mask_prob,
                'train_ratio': train_ratio
            },
            'train_history': self.train_history,
            'val_history': self.val_history,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc
        }
        
        # Uložení modelu
        if save_path:
            self.model.save_model(save_path, self.tokenizer, training_results=results)
    
        return results
    
    def test_prediction(self, test_sequence: str, num_examples: int = 5):
        """Test predikce maskovaných tokenů."""
        if not isinstance(self.model, MaskedLanguageModel):
            print("Test predikce je pouze pro masked language modely")
            return
        
        print(f"\nTest predikce ({num_examples} příkladů):")
        
        mask_token_id = self.tokenizer.vocab.get('<MASK>', 2)
        
        for i in range(num_examples):
            # Vezmi náhodný úsek
            start_pos = random.randint(0, len(test_sequence) - 100)
            test_seq = test_sequence[start_pos:start_pos + 50]
            test_tokens = self.tokenizer.tokenize(test_seq)
            
            if len(test_tokens) < 10:
                continue
            
            # Manuálně zamaskuj uprostřed
            masked_pos = len(test_tokens) // 2
            original_token = test_tokens[masked_pos]
            test_tokens[masked_pos] = '<MASK>'
            
            # Encode
            input_ids = torch.tensor([
                self.tokenizer.vocab.get(token, self.tokenizer.vocab.get('<UNK>', 1)) 
                for token in test_tokens
            ])
            
            # Predikce
            predictions, confidences = self.model.predict_masked_tokens(
                input_ids, mask_token_id, self.device
            )
            
            predicted_token_id = predictions[masked_pos].item()
            predicted_token = self.tokenizer.inverse_vocab.get(predicted_token_id, '<UNK>')
            confidence = confidences[masked_pos].item()
            
            print(f"Příklad {i+1}: {original_token} → {predicted_token} "
                  f"({'✅' if original_token == predicted_token else '❌'}) "
                  f"(conf: {confidence:.3f})")
    
    def save_results(self, results: Dict[str, Any], path: str):
        """Uloží výsledky tréninku do JSON."""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Výsledky uloženy: {path}")

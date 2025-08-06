from prepare_data import extract_chromosomes

# Nebo specifické importy
from models import MaskedLSTMGenerator, MaskedCNNGenerator, MaskedTransformerGenerator
from tokenizers import NucleotideTokenizer, KmerTokenizer
from models import DNAModelTrainer
from models import MaskedCNNGenerator
from models import MaskedTransformerGenerator

# Rychlá kontrola GPU (import torch až po models)
import torch
print(f"🔧 PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print("-" * 50)

path = "C:\\Users\\Vyrobik\\Desktop\\School\\9\\Projekt-LLM\\llm\\dna-sequence-llm\\data\\GCF_000001405.26_GRCh38_genomic.fna.gz"

chrom_seqs = extract_chromosomes(path, ["21"])

chrom_key = list(chrom_seqs.keys())[0]
chrom_seq = chrom_seqs[chrom_key]
    
print(f"Chromosome 21 sequence length: {len(chrom_seq):,} nucleotides")
print(f"Sample sequence: {chrom_seq[:50]}")

print("Nucleotide Tokenizer: ---------------------------")
nucleotideTokenizer = NucleotideTokenizer()
nucleotideTokenizer.build_vocab([chrom_seq])

print(f"vocab size: {nucleotideTokenizer.get_vocab_size()}")
print(f"vocab: {nucleotideTokenizer.get_vocab()}")

print("Masked LSTM Generator: ---------------------------")

# Vytvoření modelu
maskedLSTMGenerator = MaskedLSTMGenerator(
    vocab_size=nucleotideTokenizer.get_vocab_size(),
    embedding_dim=64,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2
)

model_info = maskedLSTMGenerator.get_model_info()
for key, value in model_info.items():
    print(f"   {key}: {value}")

# ====================== TRAINING S TRAINEREM ======================
print("\n Inicializace traineru...")

trainer = DNAModelTrainer(
    model=maskedLSTMGenerator,
    tokenizer=nucleotideTokenizer
)

# Spuštění tréninku - vše v jedné metodě!
print("\nSpouštím trénink...")
results = trainer.train(
    sequence=chrom_seq,
    num_epochs=3,           # Krátký test
    learning_rate=1e-3,
    batch_size=32,
    seq_length=256,
    mask_prob=0.15,
    train_ratio=0.8,
    save_path="trained_lstm_model.pt"
)

# Test predikce
trainer.test_prediction(chrom_seq, num_examples=5)

# Uložení výsledků
trainer.save_results(results, "training_results.json")

print(f"\n📈 Finální výsledky:")
print(f"   Train Loss: {results['final_train_loss']:.4f}")
print(f"   Val Loss: {results['final_val_loss']:.4f}")
print(f"   Train Acc: {results['final_train_acc']:.3f}")
print(f"   Val Acc: {results['final_val_acc']:.3f}")

# ====================== UKÁZKA MODULARITY ======================
print("\n🔄 Ukázka modularity - test s různými architekturami:")

print("\n1️⃣ CNN model:")

cnn_model = MaskedCNNGenerator(vocab_size=nucleotideTokenizer.get_vocab_size())
cnn_trainer = DNAModelTrainer(cnn_model, nucleotideTokenizer)
print(f"   CNN parametry: {cnn_model.count_parameters():,}")

print("\n2️⃣ Transformer model:")

transformer_model = MaskedTransformerGenerator(vocab_size=nucleotideTokenizer.get_vocab_size())
transformer_trainer = DNAModelTrainer(transformer_model, nucleotideTokenizer)
print(f"   Transformer parametry: {transformer_model.count_parameters():,}")

print("\n💡 Pro trénink jakékoliv architektury stačí:")
print("   trainer = DNAModelTrainer(model, tokenizer)")
print("   results = trainer.train(sequence, num_epochs=5)")
print("   trainer.test_prediction(test_sequence)")



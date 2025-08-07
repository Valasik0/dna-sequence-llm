import torch
import random
import numpy as np
from prepare_data import extract_chromosomes
from models import MaskedLSTMGenerator, DNAModelTrainer
from tokenizers import NucleotideTokenizer, KmerTokenizer

# Nastavení reprodukovatelnosti
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f" Seed: {SEED}")
print("-" * 50)

# ====================== 1. NAČTENÍ DAT ======================
print("Načítání chromosome 21...")
path = "C:\\Users\\Vyrobik\\Desktop\\School\\9\\Projekt-LLM\\llm\\dna-sequence-llm\\data\\GCF_000001405.26_GRCh38_genomic.fna.gz"
chrom_seqs = extract_chromosomes(path, ["21"])
chrom_key = list(chrom_seqs.keys())[0]
chrom_seq = chrom_seqs[chrom_key]

print(f"Chromosome 21 načten: {len(chrom_seq):,} nucleotides")
print(f" Ukázka: {chrom_seq[:50]}")

# ====================== 2. TOKENIZER ======================
print("🔤 Vytváření K-mer Tokenizer...")
tokenizer = KmerTokenizer(k=3)  # 3-mer tokenizer (AAA, AAT, AAG, ...)
tokenizer.build_vocab([chrom_seq])

print(f"✅ K-mer size: {tokenizer.k}")
print(f"✅ Vocab size: {tokenizer.get_vocab_size()}")
print(f"📖 Sample vocab: {list(tokenizer.get_vocab().keys())[:10]}...")  # První 10 k-merů

# ====================== 3. ROZDĚLENÍ DAT ======================
print("Rozdělení dat (train/val/test)...")
# Použijeme fixní indexy pro reprodukovatelnost
total_length = len(chrom_seq)
train_end = int(0.7 * total_length)
val_end = int(0.85 * total_length)

train_seq = chrom_seq[:train_end]
val_seq = chrom_seq[train_end:val_end]
test_seq = chrom_seq[val_end:]

print(f"Train: {len(train_seq):,} ({len(train_seq)/total_length:.1%})")
print(f"Val:   {len(val_seq):,} ({len(val_seq)/total_length:.1%})")
print(f"Test:  {len(test_seq):,} ({len(test_seq)/total_length:.1%})")

# ====================== 4. MODEL ======================
print("\n🧠 Vytváření LSTM modelu s K-mer tokenizer...")
model = MaskedLSTMGenerator(
    vocab_size=tokenizer.get_vocab_size(),
    embedding_dim=128,  # Zvětšený pro větší vocab
    hidden_dim=256,     # Zvětšený pro k-mery
    num_layers=2,
    dropout=0.2
)

model_info = model.get_model_info()
print("📊 Parametry modelu s K-mer tokenizer:")
for key, value in model_info.items():
    print(f"   {key}: {value}")

# ====================== 5. TRÉNINK ======================
print("\n🚀 Trénink LSTM modelu s K-mer tokenizer...")
trainer = DNAModelTrainer(model=model, tokenizer=tokenizer)

results = trainer.train(
    sequence=train_seq,
    num_epochs=10,
    learning_rate=1e-3,
    batch_size=32,
    seq_length=85,      # Menší kvůli k-mer tokenizaci (256/3 ≈ 85)
    mask_prob=0.15,
    save_path="lstm_kmer_model.pt"
)

# ====================== 6. TESTOVÁNÍ ======================
print("\n🔍 Testování modelu s K-mer tokenizer...")
trainer.test_prediction(test_seq, num_examples=10)

# ====================== 7. VÝSLEDKY ======================
print("\n📈 FINÁLNÍ VÝSLEDKY (LSTM + K-mer tokenizer):")
print(f"   🎯 K-mer size: {tokenizer.k}")
print(f"   📦 Vocab size: {tokenizer.get_vocab_size()}")
print(f"   🔥 Train Loss: {results['final_train_loss']:.4f}")
print(f"   📊 Val Loss:   {results['final_val_loss']:.4f}")
print(f"   ✅ Train Acc:  {results['final_train_acc']:.3f}")
print(f"   🎯 Val Acc:    {results['final_val_acc']:.3f}")

# Uložení výsledků
trainer.save_results(results, "lstm_kmer_experiment_results.json")
print("\n💾 Model a výsledky uloženy!")
print("   📄 lstm_kmer_experiment_results.json")
print("   🧠 lstm_kmer_model.pt")
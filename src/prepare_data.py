from typing import List, Dict
import gzip

def extract_chromosomes(path: str, chrom_numbers: List[str]) -> Dict[str, str]:
    """
    Extracts specified chromosome sequences from a gzipped FASTA file.

    Parameters:
        path (str): Path to the gzipped FASTA file containing chromosome sequences.
        chrom_numbers (List[str]): List of chromosome numbers or identifiers to extract (e.g., ['1', '2', 'X']).

    Returns:
        Dict[str, str]: A dictionary mapping chromosome identifiers to their cleaned DNA sequences (only A, C, G, T).

    The function reads a gzipped FASTA file, extracts only the specified chromosomes,
    removes all non-ACGT characters, and returns the sequences in uppercase.
    Chromosome identifiers are mapped from the FASTA headers to standard notation (e.g., '1', 'X', 'Y', 'MT').
    """

    chrom_targets = set(str(c) for c in chrom_numbers)
    chrom_seqs = {}
    current_id = None
    collecting = False

    chrom_id_map = {
        "NC_000001": "1",  "NC_000002": "2",  "NC_000003": "3",  "NC_000004": "4",
        "NC_000005": "5",  "NC_000006": "6",  "NC_000007": "7",  "NC_000008": "8",
        "NC_000009": "9",  "NC_000010": "10", "NC_000011": "11", "NC_000012": "12",
        "NC_000013": "13", "NC_000014": "14", "NC_000015": "15", "NC_000016": "16",
        "NC_000017": "17", "NC_000018": "18", "NC_000019": "19", "NC_000020": "20",
        "NC_000021": "21", "NC_000022": "22", "NC_000023": "X",  "NC_000024": "Y",
        "NC_012920": "MT"
    }

    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip()
                ref_id = header.split()[0][1:].split(".")[0]
                chrom_number = chrom_id_map.get(ref_id)

                collecting = chrom_number in chrom_targets
                if collecting:
                    current_id = header.split()[0][1:]
                    chrom_seqs[current_id] = []
            elif collecting:
                clean = ''.join(c for c in line.strip().upper() if c in "ACGT")
                chrom_seqs[current_id].append(clean)

    return {chrom: "".join(seq) for chrom, seq in chrom_seqs.items()}
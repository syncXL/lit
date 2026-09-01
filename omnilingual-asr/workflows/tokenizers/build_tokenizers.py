import fire
import pyarrow as pa
import pyarrow.dataset as pa_ds
from pathlib import Path
import polars as pl
import sentencepiece as spm
from collections import Counter
import csv
import random

class BuildTokenizerCLI:
    TOKENIZERS_MODELS = ["char", "bpe500", "bpe2000", "unigram500"]
    TOKENIZERS_MODELS_CONFIG = {
        "char" : {"model_type" : "char", "vocab_size" : None},
        "bpe500" : {"model_type" : "bpe", "vocab_size" : 500},
        "bpe2000" : {"model_type" : "bpe", "vocab_size": 2000},
        "unigram500" : {"model_type" : "unigram", "vocab_size" : 500}
    }

    def compile_transcript_from_dataset(self,
        parquet_dataset_root: str, output_path: str, lang_ids: list[str] = [],
        combine: bool = False, val_ratio: float = 0.2, seed: int = 123
    ):
        table = pa_ds.dataset(
            parquet_dataset_root, partitioning="hive", exclude_invalid_files=True
        ).to_table(columns=["language", "corpus", "audio_size", "text"])
        pl_table = pl.from_arrow(table.combine_chunks())

        unique_langs = pl_table["language"].unique().to_list()
        unique_langs_lower = [l.lower() for l in unique_langs]

        output_dir_root = Path(output_path).expanduser()
        output_dir_root.mkdir(parents=True, exist_ok=True)

        all_train, all_val = [], []

        for lang_name in lang_ids:
            if lang_name.lower() not in unique_langs_lower:
                print(f"{lang_name} not in {unique_langs}")
                continue

            lang_subset = pl_table.filter(pl.col("language") == lang_name)
            texts = lang_subset["text"].to_list()

            rng = random.Random(seed)
            rng.shuffle(texts)
            split_idx = int(len(texts) * (1 - val_ratio))
            train_texts, val_texts = texts[:split_idx], texts[split_idx:]

            lang_dir = output_dir_root / lang_name.lower()
            lang_dir.mkdir(parents=True, exist_ok=True)
            (lang_dir / "train.txt").write_text("\n".join(train_texts), encoding="utf-8")
            (lang_dir / "val.txt").write_text("\n".join(val_texts), encoding="utf-8")
            print(f"Saved {lang_name}: {len(train_texts)} train / {len(val_texts)} val lines")
            print(lang_dir)
            if combine:
                all_train.extend(train_texts)
                all_val.extend(val_texts)

        if combine:
            combined_dir = output_dir_root / "all_lang"
            combined_dir.mkdir(parents=True, exist_ok=True)
            (combined_dir / "train.txt").write_text("\n".join(all_train), encoding="utf-8")
            (combined_dir / "val.txt").write_text("\n".join(all_val), encoding="utf-8")
            print(f"Saved combined: {len(all_train)} train / {len(all_val)} val lines")
                
    def train_tokenizer(self,model_name : str, transcript_file : str, mdl_save_name: str):
        mdl_config = self.TOKENIZERS_MODELS_CONFIG.get(model_name, None)
        transcript_file = Path(transcript_file)

        if not transcript_file.exists():
            raise ValueError(f"File {transcript_file} doesn't exist")
        
        if not mdl_config:
            raise ValueError(f"Model {model_name} not found, choose from {self.TOKENIZERS_MODELS}")

        kwargs = dict(
            input= transcript_file,
            model_prefix=mdl_save_name,
            model_type=mdl_config["model_type"],
            character_coverage=1.0,
            normalization_rule_name="identity"
        )

        if mdl_config["vocab_size"] is not None:
            kwargs["vocab_size"] = mdl_config["vocab_size"]

        spm.SentencePieceTrainer.train(**kwargs)
        print(f"Saved {model_name} tokenizer to { mdl_save_name}.model")
    
    def analyze_tokenizer(self, model_path: str, validation_file: str) -> dict:
        """Compute compression ratio, character coverage, and casing round-trip
        rate for a trained tokenizer against a held-out validation file."""
        model_path = Path(model_path)
        validation_file = Path(validation_file)
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        unk_id = sp.unk_id()

        lines = [l for l in validation_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"No non-empty lines in {validation_file}")

        total_chars = 0
        total_tokens = 0
        casing_failures = 0
        char_counts = Counter()

        for line in lines:
            ids = sp.encode(line, out_type=int)
            total_tokens += len(ids)
            total_chars += len(line)
            char_counts.update(line)

            if sp.decode(ids) != line:
                casing_failures += 1

        compression_ratio = total_tokens / total_chars

        # character-level coverage: encode each *unique* char alone, weighted by freq
        uncovered_char_freq = sum(
            freq for ch, freq in char_counts.items()
            if unk_id in sp.encode(ch, out_type=int)
        )
        coverage = 1 - (uncovered_char_freq / total_chars)
        casing_pass_rate = 1 - (casing_failures / len(lines))

        return {
            "compression_ratio": round(compression_ratio, 4),
            "coverage": round(coverage, 4),
            "casing_pass_rate": round(casing_pass_rate, 4),
            "num_lines": len(lines),
            "num_chars": total_chars,
        }

    def _append_metrics_row(self, csv_path: str, row: dict):
        csv_path = Path(csv_path)
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def train_and_analyze(
        self,
        model_name: str,
        lang_id: str,
        train_file: str,
        validation_file: str,
        mdl_save_name: str,
        version: str,
        metrics_csv: str,
    ) -> dict:
        self.train_tokenizer(model_name, train_file, mdl_save_name)
        metrics = self.analyze_tokenizer(fr"{mdl_save_name}.model", validation_file)
        row = {"lang": lang_id, "model": model_name, "version": version, **metrics}
        self._append_metrics_row(metrics_csv, row)
        return row

    def train_all_tokenizer(self, transcript_dir: str, lang_subset: list[str], output_path: str,
                         version, model_subset: list[str] | None = None):
        model_subset = model_subset or self.TOKENIZERS_MODELS
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = output_dir / "tokenizer_metrics.csv"
        for lang_id in lang_subset:
            lang_transcript_dir = Path(transcript_dir).expanduser() / lang_id.lower()
            train_file = lang_transcript_dir / "train.txt"
            val_file = lang_transcript_dir / "val.txt"
            if not train_file.exists() or not val_file.exists():
                print(f"Missing train/val split for {lang_id}, skipping")
                continue

            lang_output_dir = output_dir / lang_id.lower()
            lang_output_dir.mkdir(parents=True, exist_ok=True)

            for model in model_subset:
                mdl_save_name = str(lang_output_dir / f"{model}_{version}")
                self.train_and_analyze(
                    model, lang_id, train_file, val_file, mdl_save_name, version, metrics_csv
                )

        
        
        
### Hf should push the datasets and tokenizers
### it isn't correctly pulling the data
### the normalizer needs to be checked

    



if __name__ == "__main__":
    # Initialize Ray if not already initialized
    fire.Fire(BuildTokenizerCLI)
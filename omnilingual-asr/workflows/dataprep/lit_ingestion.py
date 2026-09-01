import os
from functools import partial
from math import floor
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets
import fire
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds
import ray

from audio_tools import AudioTableProcessor, map_to_target_schema
from datasets import load_dataset, Audio, Value
from text_tools import normalize_text_mozilla


class FleursTextProcessor:
    """
    Batch-level processor for FLEURS text data processing.

    Handles digit replacement, text normalization, and language mapping
    for batches of FLEURS data.
    """

    def __init__(self, lang: str):
        """
        Initialize the processor for a specific language.

        Args:
            lang: Language code (e.g., "tr_tr", "ja_jp", "en_us", "fr_fr", "uk_ua")
        """
        self.lang = lang
        self.lang_mapping = {
            "tr_tr": "tur_Latn",
            "ja_jp": "jpn_Jpan",
            "en_us": "eng_Latn",
            "fr_fr": "fra_Latn",
            "uk_ua": "ukr_Cyrl",
        }

    def __call__(self, batch: pa.Table) -> pa.Table:
        # Extract transcription column as Python list
        transcriptions = batch["raw_transcription"].to_pylist()
        processed_transcriptions = []

        for text in transcriptions:
            # Normalize text
            processed_text = normalize_text_mozilla(text, is_lower=False)
            processed_transcriptions.append(processed_text)

        # Drop original transcription column and add the processed one
        batch = batch.drop(["transcription"]).append_column(
            "transcription", pa.array(processed_transcriptions, type=pa.string())
        )
        columns = ["pair_id","filename", "speaker", "lid_tokens", "lang_1", "lang_2", "duration_sec", "orthography_variant", "channel_decision", "possible_overlap", "convo_id" ]
        batch = batch.drop(columns)
        if "language" in batch.column_names:
            batch = batch.drop(["language"])

        language_values = [self.lang_mapping.get(self.lang, self.lang)] * len(batch)
        batch = batch.append_column(
            "language", pa.array(language_values, type=pa.string())
        )

        return batch

class MozillaTextProcessor:
    """
    Batch-level processor for Mozilla Text data processing
    """

    def __init__(self, lang: str):
        self.lang = lang
        self.lang_mapping = {
            "ind-jav" : "ind_jav"
        }

    
    def __call__(self, batch: pa.Table) -> pa.Table:
        transcription = batch["transcript"].to_pylist()
        processed_transcriptions = []

        for text in transcription:
            processed_text = normalize_text_mozilla(text, is_lower=False)
            processed_transcriptions.append(processed_text)
        
        batch = batch.drop(["transcript"]).append_column(
            "transcription", pa.array(processed_transcriptions, type=pa.string())
        )

        # batch = batch.drop()
        # batch = batch.drop(["lang_1", "lang_2", "pair_id"])

        language_values = [self.lang_mapping.get(self.lang, self.lang)] * len(batch)
        batch = batch.append_column(
            "language", pa.array(language_values, type=pa.string())
        )
        return batch

class DataPrepCLI:
    """Command-line interface for ASR data preparation tasks."""

    FLEURS_LAG_SUBSET = ["id_id", "jv_id"]
    MOZILLA_LAG_SUBSET = ["ind-jav"]

    # Short subset for quick testing (only 2 languages from FLEURS)
    FLEURS_SHORT_SUBSET = ["id_id", "jv_id"]

    @staticmethod
    def check_versions():
        """Check and display versions of critical packages used in data preparation.

        This helps ensure compatibility and reproducibility of the data preparation pipeline.
        """
        print("📦 Package Versions:")
        print(f"  datasets: {datasets.__version__}")
        print(f"  pyarrow:  {pa.__version__}")
        print(f"  ray:      {ray.__version__}")
        print(f"  polars:   {pl.__version__}")

        # Check for known compatibility issues
        if hasattr(datasets, "__version__"):
            datasets_ver = tuple(map(int, datasets.__version__.split(".")))
            if datasets_ver >= (3, 6, 0):
                print(
                    "⚠️  Warning: datasets version >= 3.6.0 may have compatibility issues"
                )

        if hasattr(ray, "__version__"):
            ray_ver = tuple(
                map(int, ray.__version__.split(".")[:2])
            )  # Major.minor only
            if ray_ver < (2, 49):
                print("⚠️  Warning: ray version < 2.49 may have performance issues")

    def _ingest_mls_internal(self, output_dir: str) -> None:
        """Internal method for MLS ingestion."""
        for lang in self.MLS_LANT_SUBSET:
            for split in ["test", "dev", "train"]:
                mls_hf = load_dataset(
                    "facebook/multilingual_librispeech",
                    lang,
                    split=split,
                    streaming=True,
                )
                mls_hf = mls_hf.shuffle(seed=123, buffer_size=10000)
                ray_ds_stream_ = ray.data.from_huggingface(mls_hf)

                # Use batch-level text processing
                num_cpus = max(floor((os.cpu_count() or 1) / 4), 1)
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    MLSTextProcessor,
                    fn_constructor_kwargs={"lang": lang},
                    batch_size=100,
                    batch_format="pyarrow",
                    concurrency=num_cpus,
                )

                # Audio processing
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    AudioTableProcessor,
                    fn_constructor_kwargs={
                        "audio_column": "audio.bytes",
                        "audio_format": "flac",  # or "ogg", "wav", etc.
                    },
                    batch_size=100,
                    batch_format="pyarrow",
                    concurrency=num_cpus,
                )
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    partial(map_to_target_schema, split=split, corpus="mls"),
                    batch_size=1000,
                    batch_format="pyarrow",
                )
                ray_ds_stream_.write_parquet(
                    output_dir,
                    partition_cols=["corpus", "split", "language"],
                    min_rows_per_file=10_000,
                    row_group_size=100,  # https://github.com/ray-project/ray/issues/52481
                )

    def _ingest_fleurs_internal(
        self, output_dir: str, lang_subset: list[str] | None = None
    ):
        """Internal method for FLEURS ingestion."""
        # see https://huggingface.co/datasets/google/fleurs
        # doing it on a subset of languages for simplicity

        # Use provided subset or default to full subset
        langs_to_process = (
            lang_subset if lang_subset is not None else self.FLEURS_LAG_SUBSET
        )

        split_renaming = {"validation": "dev_fleurs", "test" : "test_fleurs"}

        for lang in langs_to_process:
            for split in ["test", "validation", "train"]:
                fleurs_hf = load_dataset(
                    "google/fleurs",
                    lang,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )
                fleurs_hf = fleurs_hf.shuffle(seed=123, buffer_size=10000)
                fleurs_hf = fleurs_hf.cast_column("audio", Audio(decode=False)) 
                ray_ds_stream_ = ray.data.from_huggingface(fleurs_hf)
                
                # Use batch-level text processing
                num_cpus = max(floor((os.cpu_count() or 1) / 4), 1)
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    FleursTextProcessor,
                    fn_constructor_kwargs={"lang": lang},
                    batch_size=1000,
                    batch_format="pyarrow",
                    concurrency=num_cpus,
                )

                # Audio processing
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    AudioTableProcessor,
                    fn_constructor_kwargs={
                        "audio_column": "audio.bytes",
                        "audio_format": "flac",  # or "ogg", "wav", etc.
                    },
                    batch_size=100,
                    batch_format="pyarrow",
                    concurrency=num_cpus,
                )
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    partial(
                        map_to_target_schema,
                        split=split_renaming.get(split, split),
                        corpus="fleurs",
                    ),
                    batch_size=100,
                    batch_format="pyarrow",
                )
                ray_ds_stream_.write_parquet(
                    output_dir,
                    partition_cols=["corpus", "split", "language"],
                    min_rows_per_file=10_000,
                    row_group_size=100,  # https://github.com/ray-project/ray/issues/52481
                )

    def _ingest_custom_corpus_internal(self, output_dir: str, lang_subset: list[str] | None = None
    ) :
        langs_to_process = (
            lang_subset if lang_subset is not None else self.MOZILLA_LAG_SUBSET
        )

        split_renaming = {"val" : "train", "test" : "validation"}
        for lang in langs_to_process:
            for split in ["test", "dev", "val", "train"]:
                mozilla_hf = load_dataset(
                    "nolimitsxl/lost-in-transcription",
                    lang,
                    split=split,
                    streaming=True,
                    data_files= {split : f"{lang}/{split}-*.parquet"}
                )
                mozilla_hf = mozilla_hf.shuffle(seed=123, buffer_size=10_000)
                mozilla_hf = mozilla_hf.cast_column("speaker", Value("string"))
                mozilla_hf = mozilla_hf.cast_column("audio", Audio(decode=False)) 
                columns = ["source_dataset","filename", "speaker", "lid_tokens", "lang_1", "lang_2", "duration_sec", "orthography_variant", "channel_decision", "possible_overlap","convo_id"]
                mozilla_hf = mozilla_hf.remove_columns(columns)
                ray_ds_stream_ = ray.data.from_huggingface(mozilla_hf)

                num_cpus = max(floor((os.cpu_count() or 1) / 4), 1)
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    MozillaTextProcessor,
                    fn_constructor_kwargs={"lang" : lang},
                    batch_size=1000,
                    batch_format="pyarrow",
                    concurrency=num_cpus
                )

                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    AudioTableProcessor,
                    fn_constructor_kwargs={
                        "audio_column" : "audio.bytes",
                        "audio_format" : "flac",
                    },
                    batch_size=100,
                    batch_format="pyarrow",
                    concurrency=num_cpus
                )

                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    partial(
                        map_to_target_schema,
                        split=split_renaming.get(split, split),
                        corpus="mozilla"
                    ),
                    batch_size=100,
                    batch_format="pyarrow"
                )
                ray_ds_stream_.write_parquet(
                    output_dir,
                    partition_cols=["corpus", "split", "language"],
                    min_rows_per_file=10_000,
                    row_group_size=100,
                )

    @staticmethod
    def _compute_distribution_stats_internal(
        parquet_dataset_root: str, output_path: str
    ):
        """Internal method for computing distribution statistics."""
        table = pa_ds.dataset(
            parquet_dataset_root, partitioning="hive", exclude_invalid_files=True
        ).to_table(columns=["language", "corpus", "audio_size"])
        pl_table = pl.from_arrow(table.combine_chunks())
        assert isinstance(pl_table, pl.DataFrame)
        stats = pl_table.group_by(["corpus", "language"]).agg(
            (pl.col("audio_size").sum() / 3600 / 16_000).alias("hours")
        )
        stats.write_csv(output_path, separator="\t")
        return output_path

    def ingest_mls(self, output_dir: str):
        """Ingest Multilingual LibriSpeech (MLS) datasets.

        Args:
            output_dir: Output directory path for processed Parquet files
        """
        print(f"Starting MLS ingestion to: {output_dir}")
        self._ingest_mls_internal(output_dir)
        print("MLS ingestion completed")

    def ingest_fleurs(self, output_dir: str, lang_subset:list[str] | None = None):
        """Ingest FLEURS datasets.

        Args:
            output_dir: Output directory path for processed Parquet files
            lang_subset: The language subset to process

        """
        print(f"Starting FLEURS ingestion to: {output_dir}")
        self._ingest_fleurs_internal(output_dir, lang_subset)
        print("FLEURS ingestion completed")

    def ingest_custom_corpus(self, output_dir: str, lang_subset: list[str] | None = None):
        """Ingest Mozilla datasets

        Args:
            output_dir: Output directory path for processed Parquet files
            lang_subset: The language subset to process
        """

        print(f"Starting Mozilla Ingesstion to: {output_dir}")
        self._ingest_custom_corpus_internal(output_dir, lang_subset)
        print("Mozilla ingestion completed")
      
    def compute_stats(self, parquet_dataset_root: str, output_path: str):
        """Compute distribution statistics from processed datasets.

        Args:
            parquet_dataset_root: Path to the root of partitioned Parquet dataset
            output_path: Output path for TSV statistics file
        """
        print(f"Computing stats for: {parquet_dataset_root}")
        result_path = self._compute_distribution_stats_internal(
            parquet_dataset_root, output_path
        )
        print(f"Statistics saved to: {result_path}")
        return result_path

    def test_dataset(self, dataset_path: str, **kwargs):
        """
        Test dataset functionality - redirects to dedicated dataloader_example module.

        Args:
            dataset_path: Path to the dataset directory
            **kwargs: Additional arguments passed to dataloader_example
        """
        print("📚 For dataset testing, use the dedicated dataloader_example module:")
        print(
            f"   python -m omnilingual_asr.dataprep.dataloader_example test_dataset --dataset_path='{dataset_path}'"
        )
        print("\n🔧 Available method:")
        print("   • test_dataset: Basic dataset testing with iterations")

        from dataloader_example import DataLoaderExample

        loader = DataLoaderExample()
        return loader.test_dataset(dataset_path, **kwargs)

    def run_short(
        self, output_dir: str, name: str = "all_asr_short", version: str = "0"
    ):
        """Run short data preparation pipeline (only 2 languages from FLEURS for quick testing).

        Args:
            output_dir: Base output directory path
            name: Dataset name (default: "all_asr_short")
            version: Dataset version (default: "0")
        """
        print("🚀 Starting SHORT data preparation pipeline")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Dataset name: {name}, Version: {version}")
        print(
            f"🌍 Processing only {len(self.FLEURS_SHORT_SUBSET)} languages from FLEURS: {self.FLEURS_SHORT_SUBSET}"
        )

        parquet_dataset_root = str(Path(output_dir) / f"{name}/version={version}/")

        # Only ingest FLEURS with short subset (no MLS for speed)
        print("🔄 Ingesting FLEURS with short language subset...")
        self._ingest_fleurs_internal(
            parquet_dataset_root, lang_subset=self.FLEURS_SHORT_SUBSET
        )

        self._ingest_custom_corpus_internal(
            parquet_dataset_root, lang_subset=self.MOZILLA_LAG_SUBSET
        )

        # Compute statistics
        stats_path = Path(output_dir) / f"{name}/language_distribution_{version}.tsv"
        self.compute_stats(parquet_dataset_root, str(stats_path))

        print("✅ SHORT pipeline finished successfully!")
        print(f"📈 Dataset ready at: {parquet_dataset_root}")
        print(f"📊 Statistics saved at: {stats_path}")

        # Test the dataset
        self.test_dataset(parquet_dataset_root, stats_path=stats_path, num_iterations=5)
        return parquet_dataset_root, stats_path

    def run_full(self, output_dir: str, name: str = "all_asr", version: str = "0"):
        """Run complete data preparation pipeline (MLS + full FLEURS + stats).

        Args:
            output_dir: Base output directory path
            name: Dataset name (default: "all_asr")
            version: Dataset version (default: "0")
        """
        print("🚀 Starting FULL data preparation pipeline")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Dataset name: {name}, Version: {version}")
        print(
            f"🌍 Processing {len(self.FLEURS_LAG_SUBSET)} languages from FLEURS: {self.FLEURS_LAG_SUBSET}"
        )
        print(
            f"📚 Processing {len(self.MLS_LANT_SUBSET)} languages from MLS: {self.MLS_LANT_SUBSET}"
        )

        parquet_dataset_root = str(Path(output_dir) / f"{name}/version={version}/")

        # Ingest both datasets
        print("🔄 Ingesting MLS datasets...")
        self.ingest_mls(parquet_dataset_root)
        print("🔄 Ingesting FLEURS datasets...")
        self.ingest_fleurs(parquet_dataset_root)

        # Compute statistics
        stats_path = Path(output_dir) / f"{name}/language_distribution_{version}.tsv"
        self.compute_stats(parquet_dataset_root, str(stats_path))

        print("✅ FULL pipeline finished successfully!")
        print(f"📈 Dataset ready at: {parquet_dataset_root}")
        print(f"📊 Statistics saved at: {stats_path}")

        # Test the dataset
        self.test_dataset(parquet_dataset_root, stats_path=stats_path)
        return parquet_dataset_root, stats_path

    def run_set(self, output_dir: str, lang_set_id: str, name: str ="all_asr", version: str = "0"):
        # CS: Moz, P: F/V
        set_mapping = {
            "ind-jav" : {
                "FLEURS" : ("id_id", "jv_id"),
            }
        }

        lang_set = set_mapping.get(lang_set_id, None)
        if not lang_set:
            raise ValueError("No LAng set found")
            
        fleurs_lang_subset = lang_set.get("FLEURS")
        custom_lang_subset = [lang_set_id]
        
        
        print("🚀 Starting FULL data preparation pipeline")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Dataset name: {name}, Version: {version}")
        print(
            f"🌍 Processing {len(fleurs_lang_subset)} languages from FLEURS: {fleurs_lang_subset}"
        )
        print(
            f"📚 Processing {len(custom_lang_subset)} languages from Custom: {custom_lang_subset}"
        )

        parquet_dataset_root = str(Path(output_dir) / f"{name}/version={version}/")

        print("🔄 Ingesting Custom datasets...")
        self.ingest_custom_corpus(parquet_dataset_root,custom_lang_subset)

        print("🔄 Ingesting FLEURS datasets...")
        self.ingest_fleurs(parquet_dataset_root,fleurs_lang_subset)

        
        # Compute statistics
        stats_path = Path(output_dir) / f"{name}/language_distribution_{version}.tsv"
        self.compute_stats(parquet_dataset_root, str(stats_path))

        print("✅ FULL pipeline finished successfully!")
        print(f"📈 Dataset ready at: {parquet_dataset_root}")
        print(f"📊 Statistics saved at: {stats_path}")

        # Test the dataset
        self.test_dataset(parquet_dataset_root, stats_path=stats_path)
        return parquet_dataset_root, stats_path

    def normalizer(self,):
        pass

if __name__ == "__main__":
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    try:
        fire.Fire(DataPrepCLI)
    finally:
        # Clean shutdown of Ray
        if ray.is_initialized():
            ray.shutdown()

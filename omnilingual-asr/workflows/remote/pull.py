import fire

from huggingface_hub import snapshot_download
from pathlib import Path

class PullCli:
    HF_REPO_ID = "nolimitsxl"
    DATASETS_REPO = "lost-in-transcription-prepared"
    MAIN_REPO = "lost-in-transcription-artifacts"
    DIR = "."

    def pull_dataset(self, ds_repo_id: str | None = None, ds_dir: str | None = None,
                      revision: str | None = None):
        ds_repo_id = ds_repo_id or f"{self.HF_REPO_ID}/{self.DATASETS_REPO}"
        ds_dir = Path(ds_dir) if ds_dir else Path(self.DIR) / "data"
        snapshot_download(
            repo_id=ds_repo_id,
            repo_type="dataset",
            local_dir=str(ds_dir),
            revision=revision,
        )
        print(f"Pulled dataset -> {ds_dir}")

    def pull_tokenizers(self, repo_id: str | None = None, artifacts_dir: str | None = None,
                         revision: str | None = None):
        repo_id = repo_id or f"{self.HF_REPO_ID}/{self.MAIN_REPO}"
        artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path(self.DIR) / "artifacts"
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(artifacts_dir),
            allow_patterns=["tokenizers/*"],
            revision=revision,
        )
        print(f"Pulled tokenizers -> {artifacts_dir}/tokenizers")

    def pull_models(self, repo_id: str | None = None, artifacts_dir: str | None = None,
                     revision: str | None = None):
        repo_id = repo_id or f"{self.HF_REPO_ID}/{self.MAIN_REPO}"
        artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path(self.DIR) / "artifacts"
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(artifacts_dir),
            revision=revision,
        )
        print(f"Pulled models -> {artifacts_dir}/models")

    def pull_all(self, ds_repo_id: str | None = None, file_repo_id: str | None = None,
                 ds_dir: str | None = None, artifacts_dir: str | None = None,
                 revision: str | None = None):
        self.pull_dataset(ds_repo_id, ds_dir, revision)
        self.pull_tokenizers(file_repo_id, artifacts_dir, revision)
        self.pull_models(file_repo_id, artifacts_dir, revision)
        print("All Pulls Successful")

if __name__ == "__main__":
    # Initialize Ray if not already initialized
    fire.Fire(PullCli)
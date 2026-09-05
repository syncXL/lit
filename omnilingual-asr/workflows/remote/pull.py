import fire
import re

from huggingface_hub import snapshot_download, HfApi
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
                 revision: str | None = None, keep_last_n_steps: int = 2):
        repo_id = repo_id or f"{self.HF_REPO_ID}/{self.MAIN_REPO}"
        artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path(self.DIR) / "artifacts"

        api = HfApi()
        repo_files = api.list_repo_files(repo_id, repo_type="model", revision=revision)

        step_pattern = re.compile(r"(?:^|/)step_(\d+)/")
        step_nums = sorted({int(m.group(1)) for f in repo_files if (m := step_pattern.search(f))})

        if not step_nums:
            print(f"No step_N checkpoints found under {repo_id}, pulling everything")
            allow_patterns = None
        else:
            keep_steps = step_nums[-keep_last_n_steps:]
            print(f"Found steps {step_nums}, keeping {keep_steps}")

            # match any file NOT under a step_N/ dir, plus files under the kept step dirs
            all_step_dirs = {f"step_{n}" for n in step_nums}
            kept_step_dirs = {f"step_{n}" for n in keep_steps}

            allow_patterns = ["*"]  # baseline: everything not under a step_N dir
            for step_dir in kept_step_dirs:
                allow_patterns.append(f"{step_dir}/*")
                allow_patterns.append(f"*/{step_dir}/*")  # in case checkpoints/ nests it

            ignore_patterns = []
            for step_dir in all_step_dirs - kept_step_dirs:
                ignore_patterns.append(f"{step_dir}/*")
                ignore_patterns.append(f"*/{step_dir}/*")

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(artifacts_dir),
            revision=revision,
            allow_patterns=allow_patterns if step_nums else None,
            ignore_patterns=ignore_patterns if step_nums else None,
        )
        print(f"Pulled models -> {artifacts_dir}")

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
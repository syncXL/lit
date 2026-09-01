import fire

from huggingface_hub import HfApi
from pathlib import Path


class CommitCli:
    HF_REPO_ID = "nolimitsxl"
    DATASETS_REPO = "lost-in-transcription-prepared"
    MAIN_REPO = "lost-in-transcription-artifacts"
    DIR = "."

    def init_api(self):
        return HfApi()

    def init_repos(self):
        """Run once, manually, when setting up a new project. Not in __init__."""
        api = self.init_api()
        api.create_repo(f"{self.HF_REPO_ID}/{self.DATASETS_REPO}", repo_type="dataset", exist_ok=True)
        api.create_repo(f"{self.HF_REPO_ID}/{self.MAIN_REPO}", repo_type="model", exist_ok=True)
        print("Repos ready")

    def push_dataset(self, ds_repo_id: str | None = None, ds_dir: str | None = None):
        ds_repo_id = ds_repo_id or f"{self.HF_REPO_ID}/{self.DATASETS_REPO}"
        ds_dir = Path(ds_dir) if ds_dir else Path(self.DIR) / "data"
        if not ds_dir.exists():
            raise FileNotFoundError(f"{ds_dir} does not exist")
        api = self.init_api()
        api.create_repo(ds_repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=ds_repo_id, repo_type="dataset", folder_path=str(ds_dir))
        print("Upload Dataset Successful")

    def push_tokenizers(self, repo_id: str | None, tkn_dir: str | None = None):
        repo_id = repo_id or f"{self.HF_REPO_ID}/{self.MAIN_REPO}"
        tkn_dir = Path(tkn_dir) if tkn_dir else Path(self.DIR) / "files/tokenizers"
        if not tkn_dir.exists():
            raise FileNotFoundError(f"{tkn_dir} does not exist")

        api = self.init_api()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(tkn_dir), path_in_repo="tokenizers/")
        print("Upload Tokenizers Successfull")
    
    def push_models(self, repo_id: str | None, model_dir: str | None = None):
        repo_id = repo_id or f"{self.HF_REPO_ID}/{self.MAIN_REPO}"
        model_dir = Path(model_dir) if model_dir else Path(self.DIR) / f"files/models"

        if not model_dir.exists():
            raise FileNotFoundError("Folder does not exist")
        
        api = self.init_api()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(model_dir), path_in_repo="models/")
        print("Upload Model Successfull")

    def commit_all(self, ds_repo_id: str | None = None, file_repo_id: str | None = None,
               model_dir: str | None = None, tkn_dir: str | None = None, ds_dir: str | None = None):
        self.push_dataset(ds_repo_id, ds_dir)
        self.push_tokenizers(file_repo_id, tkn_dir)
        self.push_models(file_repo_id, model_dir)
        print("All Uploads Successful")
    
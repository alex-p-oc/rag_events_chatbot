"""
Model configuration module
"""

from huggingface_hub import snapshot_download

# HuggingFace model name used for generating embeddings
# - Good balance between accuracy and performance
# - Suitable for semantic search and RAG pipelines
#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME = "./models/all-mpnet-base-v2"

def download_local_model():
    """
    Download HuggingFace model locally
    """
    local_path = snapshot_download(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        local_dir=MODEL_NAME
    )
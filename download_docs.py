import os
import gdown

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    
    files = {
        "docs/doc1.pdf": "1oWcyH0XkzpHeWozMBWJSFEUEw70Lrc2-",
        "docs/doc2.pdf": "1m1SudlRSlEK7y_-jweDjhPB5VVWzmQ7-",
        "docs/doc3.pdf": "1suFO8EBLxRH6hKKcJln4a9PRsOGu2oYj"
    }
    
    for path, file_id in files.items():
        if not os.path.exists(path):
            print(f"Downloading {file_id} to {path}...")
            gdown.download(id=file_id, output=path, quiet=False)
        else:
            print(f"{path} already exists")

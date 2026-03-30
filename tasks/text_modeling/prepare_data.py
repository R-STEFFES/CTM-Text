import os
import requests

def download_tinyshakespeare(data_dir):
    """Downloads the TinyShakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = os.path.join(data_dir, "tinyshakespeare.txt")
    
    if os.path.exists(file_path):
        print(f"Dataset already exists at {file_path}")
        return file_path
        
    print(f"Downloading TinyShakespeare to {file_path}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
        
    print("Download complete.")
    return file_path

if __name__ == "__main__":
    # Create data directory inside tasks/text_modeling
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    download_tinyshakespeare(data_dir)

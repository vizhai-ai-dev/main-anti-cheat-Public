import os
import urllib.request
import sys

def download_file(url, filename):
    """Download file from URL with progress indicator"""
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping download.")
        return True
    
    print(f"Downloading {filename}...")
    print(f"Source: {url}")
    
    def report_progress(count, block_size, total_size):
        if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r{percent}% completed")
        sys.stdout.flush()
    
    try:
        # Add headers to avoid 403 errors
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            with open(filename, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        sys.stdout.write(f"\r{percent}% completed")
                        sys.stdout.flush()
        
        print(f"\nDownload of {filename} completed successfully.")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Remove partial file
        return False

def main():
    # Create directory for models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Try downloading YOLOv3 weights from a reliable source
    print("Downloading YOLOv3 weights...")
    
    # Use wget-like approach for the weights file
    weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights'
    
    if not download_file(weights_url, 'yolov3.weights'):
        print("Primary source failed. Trying alternative...")
        # Alternative approach - download from a mirror
        alt_url = 'https://pjreddie.com/media/files/yolov3.weights'
        if not download_file(alt_url, 'yolov3.weights'):
            print("\nAll download attempts failed. Please download manually from:")
            print("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights")
            print("or")
            print("https://pjreddie.com/media/files/yolov3.weights")
            print("and place it in the current directory.")
    
    # Download YOLOv3 config
    download_file(
        'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'yolov3.cfg'
    )
    
    # Download COCO names
    download_file(
        'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
        'coco.names'
    )
    
    print("Model download process completed.")

if __name__ == "__main__":
    main() 
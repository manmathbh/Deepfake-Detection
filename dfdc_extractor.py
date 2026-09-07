import os
import zipfile
import urllib.request
import subprocess

# 1. Fetch the required video list
print("Fetching target video list...")
urllib.request.urlretrieve("https://raw.githubusercontent.com/ahaliassos/RealForensics/main/data/DFDC/dfdc_vids.txt", "dfdc_vids.txt")

with open('dfdc_vids.txt', 'r') as f:
    req_vids = set(os.path.basename(line.strip()).replace('.mp4', '') for line in f if line.strip())

output_archive = 'DFDC_Evaluation_Subset.zip'
found_count = 0

print(f"Targeting {len(req_vids)} specific videos. Starting rolling download-extraction...")

with zipfile.ZipFile(output_archive, 'a', zipfile.ZIP_DEFLATED) as out_zip:
    # 2. Iterate through chunks 20 to 49
    for i in range(20, 50):
        chunk = f"{i:02d}.zip"
        print(f"\n--- Processing {chunk} ---")
        
        # Download the specific chunk via Kaggle API
        try:
            subprocess.run(["kaggle", "competitions", "download", "-c", "deepfake-detection-challenge", "-f", chunk], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to download {chunk}. Skipping...")
            continue
            
        # Scan and extract matches
        print(f"Scanning inside {chunk}...")
        try:
            with zipfile.ZipFile(chunk, 'r') as z:
                matches = [m for m in z.namelist() if os.path.basename(m).replace('.mp4', '') in req_vids]
                
                for m in matches:
                    out_zip.writestr(os.path.basename(m), z.read(m))
                    found_count += 1
                    
            print(f"Extracted {len(matches)} matches. Total captured: {found_count}")
        except zipfile.BadZipFile:
            print(f"Error: {chunk} is corrupted.")
            
        # Obliterate the chunk to keep your storage free
        if os.path.exists(chunk):
            os.remove(chunk)
            print(f"Deleted {chunk} from local disk.")

print(f"\nPipeline complete! Successfully packaged {found_count} videos into {output_archive}.")

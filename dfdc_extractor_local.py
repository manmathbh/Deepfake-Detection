import os
import zipfile
import urllib.request

# 1. Fetch the required video list if it doesn't exist yet
if not os.path.exists("dfdc_vids.txt"):
    print("Fetching target video list...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ahaliassos/RealForensics/main/data/DFDC/dfdc_vids.txt", "dfdc_vids.txt")

with open('dfdc_vids.txt', 'r') as f:
    req_vids = set(os.path.basename(line.strip()).replace('.mp4', '') for line in f if line.strip())

output_archive = 'DFDC_Evaluation_Subset.zip'
found_count = 0
download_dir = "./" # Directory where your manually downloaded zips are located

print(f"Targeting {len(req_vids)} specific videos. Starting local extraction...")

with zipfile.ZipFile(output_archive, 'a', zipfile.ZIP_DEFLATED) as out_zip:
    existing_files = set(out_zip.namelist())
    
    # 2. Iterate through chunks 20 to 49
    for i in range(20, 50):
        chunk_name = f"{i:02d}.zip"
        chunk_path = os.path.join(download_dir, chunk_name)
        print(f"\n--- Processing {chunk_path} ---")
        
        # Check if the file exists locally
        if not os.path.exists(chunk_path):
            print(f"File {chunk_name} not found locally. Please download it manually from Kaggle. Skipping...")
            continue
            
        # Scan and extract matches
        print(f"Scanning inside {chunk_name}...")
        try:
            with zipfile.ZipFile(chunk_path, 'r') as z:
                matches = [m for m in z.namelist() if os.path.basename(m).replace('.mp4', '') in req_vids]
                
                extracted_this_chunk = 0
                for m in matches:
                    file_basename = os.path.basename(m)
                    if file_basename not in existing_files:
                        out_zip.writestr(file_basename, z.read(m))
                        existing_files.add(file_basename)
                        found_count += 1
                        extracted_this_chunk += 1
                        
                print(f"Extracted {extracted_this_chunk} matches from this chunk. Total captured so far: {found_count}")
        except zipfile.BadZipFile:
            print(f"Error: {chunk_name} is corrupted.")
            
        # Optional: Ask if you want to delete the chunk after processing to save space
        # if os.path.exists(chunk_path):
        #     os.remove(chunk_path)
        #     print(f"Deleted {chunk_name} from local disk to save space.")

print(f"\nPipeline complete! Successfully packaged {found_count} new videos into {output_archive}.")

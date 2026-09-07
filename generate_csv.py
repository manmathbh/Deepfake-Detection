import os
import csv
import random
import json

root_dir = "/home/manmath/OpenAVFF/dfdc_train_part_0"
metadata_path = os.path.join(root_dir, "metadata.json")

train_csv = "data/trainset.csv"
val_csv = "data/valset.csv"

print("Loading metadata.json...")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

real_videos = []
fake_videos = []

for filename, info in metadata.items():
    if filename.endswith(".mp4"):
        full_path = os.path.join(root_dir, filename)
        
        if os.path.exists(full_path):
            if info["label"] == "REAL":
                real_videos.append([full_path, 0])
            else:
                fake_videos.append([full_path, 1])

# --- UNDERSAMPLING FAKE DATA ---
# Limit the fake videos to 200 to balance against the 86 real videos
random.seed(42) 
fake_videos = random.sample(fake_videos, min(200, len(fake_videos)))

all_data = real_videos + fake_videos
random.shuffle(all_data)

split_index = int(len(all_data) * 0.8)
train_data = all_data[:split_index]
val_data = all_data[split_index:]

with open(train_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "target"])
    writer.writerows(train_data)

with open(val_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "target"])
    writer.writerows(val_data)

print(f"Total videos used: {len(all_data)} ({len(real_videos)} Real, {len(fake_videos)} Fake)")
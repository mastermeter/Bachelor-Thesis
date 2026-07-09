# This files was run without the usage of slurm job/files 

import os
import csv
import requests
import time

from dotenv import load_dotenv

from load_swiss_topo import download_swisstopo_aerial_tile
from mapillary_sdk import get_filtered_panoramic_features

load_dotenv()

MAPILLARY_TOKEN = os.environ["MAPILLARY_TOKEN"]


GEOJSON_INPUT="mp_aigle.geojson"
DATASET_DIR = "swiss_dataset"

os.makedirs(f"{DATASET_DIR}/ground", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/aerial", exist_ok=True)

# From the list of link returned, download the related images
def get_image_download_url(image_id, token):
    url = f"https://graph.mapillary.com/{image_id}"
    headers = {"Authorization": f"OAuth {token}"}
    params = {"fields": "thumb_2048_url"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code == 200:
            return response.json().get("thumb_2048_url")
        elif response.status_code == 429:
            print(f" [!] Rate limit (429)")
            time.sleep(10)
        return None
    except Exception:
        return None

def main():
    print("[*] Step 1 : Street-view images...")
    panoramic_features = get_filtered_panoramic_features(GEOJSON_INPUT, MAPILLARY_TOKEN)
    
    if not panoramic_features:
        print("[-] Nothing returned")
        return
        
    print(f"[+] {len(panoramic_features)} found, filtering and pairing")

    csv_path = f"{DATASET_DIR}/metadata.csv"
    file_exists = os.path.exists(csv_path)

    #Avoid duplication of existing images
    existing_images = set()
    if file_exists:
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            next(reader,None)
            for row in reader:
                existing_images.add(row[0])
    
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["image_id", "lat", "lon", "heading", "ground_path", "aerial_path"])

        success_count = 0
        total_features = len(panoramic_features)
        
        for index, feature in enumerate(panoramic_features):
            props = feature.get("properties", {})
            img_id = str(props.get("id"))
            heading = props.get("compass_angle")
            lon, lat = feature.get("geometry", {}).get("coordinates", [None, None])

            print(f" -> Processing [{index + 1}/{total_features}] | ID: {img_id}")

            if img_id in existing_images:
                print(f"[-] Skipping image {img_id} : Already existing")
                continue

            if heading is None:
                print(f"[-] Image {img_id} : Missing orientation ")
                continue

            img_url = get_image_download_url(img_id, MAPILLARY_TOKEN)
            if not img_url:
                continue

            try:
                ground_csv_path = f"ground/{img_id}.jpg"
                aerial_csv_path = f"aerial/{img_id}.jpg"

                ground_disk_path = os.path.join(DATASET_DIR, ground_csv_path)
                ground_resp = requests.get(img_url, timeout=15)
                ground_resp.raise_for_status()

                with open(ground_disk_path, "wb") as f:
                    f.write(ground_resp.content)

                aerial_success = download_swisstopo_aerial_tile(lat, lon, img_id, DATASET_DIR)

                if aerial_success:
                    writer.writerow([img_id, lat, lon, heading, ground_csv_path, aerial_csv_path])
                    print(f"[+] Pair {img_id} formed : (Angle: {heading}°)")
                    success_count += 1
                else:
                    if os.path.exists(ground_disk_path):
                        os.remove(ground_disk_path)

            except Exception as e:
                print(f"[-] Error during : {img_id} : {e}")

    print(f"\n[=] End : {success_count} pairs generated '{DATASET_DIR}/'.")
    print(f"[=] Index file : {csv_path}")


if __name__ == "__main__":
    main()
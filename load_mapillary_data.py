import os
import csv
import requests

from dotenv import load_dotenv

from load_swiss_topo import download_swisstopo_aerial_tile

load_dotenv()

MAPILLARY_TOKEN = os.environ["MAPILLARY_TOKEN"]

BBOX = [7.420, 46.944, 7.452, 46.950]

DATASET_DIR = "swiss_dataset"

os.makedirs(f"{DATASET_DIR}/ground", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/aerial", exist_ok=True)


def fetch_mapillary_data(bbox, token, limit=1):
    url = "https://graph.mapillary.com/images"
    headers = {"Authorization": f"OAuth {token}"}
    params = {
        "bbox": ",".join(map(str, bbox)),
        "fields": "id,thumb_2048_url,geometry,compass_angle,camera_type",
        "limit": limit,
        "camera_type": "panoramic"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"[-] Error : {e}")
        return []

def main():
    print("[*] Step 1 : Street-view images...")
    mapillary_results = fetch_mapillary_data(BBOX, MAPILLARY_TOKEN, limit=1000)
    
    if not mapillary_results:
        print("[-] Nothing returned")
        return
        
    print(f"[+] {len(mapillary_results)} found, filtering and pairing")

    csv_path = f"{DATASET_DIR}/metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_id", "lat", "lon", "heading", "ground_path", "aerial_path"])

        success_count = 0

        for item in mapillary_results:
            img_id = item["id"]
            img_url = item["thumb_2048_url"]
            heading = item.get("compass_angle")
            camera_type = item.get("camera_type")
            lon, lat = item["geometry"]["coordinates"]

            if camera_type != "panoramic":
                print(f"[-] Image {img_id} : Not a 360° view")

            if heading is None:
                print(f"[-] Image {img_id} : Missing orientation ")
                continue

            try:
                ground_resp = requests.get(img_url, timeout=15)
                ground_resp.raise_for_status()
                ground_path = f"{DATASET_DIR}/ground/{img_id}.jpg"
                with open(ground_path, "wb") as f:
                    f.write(ground_resp.content)

                aerial_success = download_swisstopo_aerial_tile(lat, lon, img_id, DATASET_DIR)

                if aerial_success:
                    aerial_path = f"{DATASET_DIR}/aerial/{img_id}.jpg"
                    writer.writerow([img_id, lat, lon, heading, ground_path, aerial_path])
                    print(f"[+] Pair {img_id} formed : (Angle: {heading}°)")
                    success_count += 1
                else:
                    if os.path.exists(ground_path):
                        os.remove(ground_path)

            except Exception as e:
                print(f"[-] Erreur lors du traitement de l'image {img_id} : {e}")

    print(f"\n[=] Fin du traitement. {success_count} paires valides enregistrées dans '{DATASET_DIR}/'.")
    print(f"[=] Fichier d'indexation généré : {csv_path}")


if __name__ == "__main__":
    main()
import json
import mapillary as mly

def get_filtered_panoramic_features(geojson_path, token):
    mly.interface.set_access_token(token)
    
    try:
        with open(geojson_path, mode='r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        print(f"[-] Error : File '{geojson_path}' missing")
        return []

    raw_data = mly.interface.images_in_shape(geojson_data)
    data_dict = raw_data.to_dict()

    features = data_dict.get("features", [])

    panoramic_features = [
        f for f in features 
        if f.get("properties", {}).get("is_pano") is True
    ]
    print(f"[+] Filtering done : {len(panoramic_features)} 360-views")
    
    return panoramic_features
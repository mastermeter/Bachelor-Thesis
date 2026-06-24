from pyproj import Transformer
import requests

def download_swisstopo_aerial_tile(lat, lon, image_id, dataset_dir, size_meters=100, pixels=640):

    # 1. Conversion WGS84 (GPS mondial) -> LV95 (Projection suisse CH1903+)
    transformer = Transformer.from_crs("epsg:4326", "epsg:2056", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    
    half_size = size_meters / 2
    bbox_lv95 = [
        easting - half_size,  
        northing - half_size, 
        easting + half_size,  
        northing + half_size
    ]
    
    url = "https://wms.geo.admin.ch/"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ch.swisstopo.swissimage",  
        "STYLES": "",
        "CRS": "EPSG:2056",                   
        "BBOX": ",".join(map(str, bbox_lv95)),
        "WIDTH": str(pixels),                 
        "HEIGHT": str(pixels),
        "FORMAT": "image/jpeg"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            if b"Exception" in response.content or b"xml" in response.content[:100].lower():
                print(f"[-] Server Error for :  {image_id}")
                return False
                
            path = f"{dataset_dir}/aerial/{image_id}.jpg"
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"[-] Error HTTP {response.status_code}, ID : {image_id}")
            return False
            
    except Exception as e:
        print(f"[-] Network Error : {image_id} : {e}")
        return False
import os
import requests
import base64

def save_debug(fname, prefix):
    url = 'http://127.0.0.1:5051/api/analyze/debug'
    file_path = os.path.join('data set', fname)
    
    with open(file_path, 'rb') as f:
        res = requests.post(url, files={'image': f}).json()
        
    imgs = res['images']
    
    with open(f'{prefix}_annotated.png', 'wb') as f:
        f.write(base64.b64decode(imgs['annotated_result']))
    with open(f'{prefix}_forehead.png', 'wb') as f:
        f.write(base64.b64decode(imgs['forehead_mask']))
    with open(f'{prefix}_canthus.png', 'wb') as f:
        f.write(base64.b64decode(imgs['inner_canthus_mask']))
        
    print(f"Successfully saved debug images for {fname}:")
    print(f"  - {prefix}_annotated.png")
    print(f"  - {prefix}_forehead.png")
    print(f"  - {prefix}_canthus.png")

if __name__ == "__main__":
    save_debug('101_1_2_1_101_100_1.png', 'roi_debug_101_1_2_1')
    save_debug('101_1_2_3_111_101_1.png', 'roi_debug_101_1_2_3')

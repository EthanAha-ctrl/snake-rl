import os
import lmdb
import pickle
try:
    import bson
except ImportError:
    print("Error: 'bson' module is missing. Please install it using 'pip install pymongo'")
    exit(1)

DATA_ROOT = os.path.join("reference", "project7", "data")
INPUT_LMDB = os.path.join(DATA_ROOT, "coc_train.lmdb")
INPUT_META = os.path.join(DATA_ROOT, "coc_meta_blur.pkl") # Usually preprocessing writes to coc_meta.pkl

# Double check if meta is named coc_meta.pkl or something else.
# reference/project7/preprocessing.py outputs to coc_train.lmdb and coc_meta.pkl
INPUT_META = os.path.join(DATA_ROOT, "coc_meta.pkl")
OUTPUT_BSON = "coc_images.bson"

def convert_lmdb_to_bson():
    if not os.path.exists(INPUT_LMDB):
        print(f"Error: {INPUT_LMDB} not found. Ensure preprocessing.py was run to generate the raw images LMDB.")
        return

    print("Loading metadata...")
    with open(INPUT_META, 'rb') as f:
         meta_info = pickle.load(f)
         
    print(f"Found {len(meta_info)} entries in metadata.")
    
    env = lmdb.open(INPUT_LMDB, readonly=True, lock=False)
    
    bson_file = open(OUTPUT_BSON, 'wb')
    count = 0
    with env.begin() as txn:
        for entry in meta_info:
            if len(entry) == 3:
                key_str, r, sharpness_grid = entry
            else:
                key_str, r = entry
                sharpness_grid = None
            
            val_bytes = txn.get(key_str.encode('ascii'))
            if val_bytes is None:
                continue
                
            doc = {
                "key": key_str,
                "label_r": r,
                "image_png": val_bytes
            }
            if sharpness_grid is not None:
                doc["sharpness_grid"] = sharpness_grid.tobytes()
                
            # Write a single BSON document (the file will be a concatenated sequence of BSON objects)
            bson_file.write(bson.BSON.encode(doc))
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} images...")
                
    bson_file.close()
    env.close()
    print(f"Done! Saved {count} records to {OUTPUT_BSON}")

if __name__ == "__main__":
    convert_lmdb_to_bson()

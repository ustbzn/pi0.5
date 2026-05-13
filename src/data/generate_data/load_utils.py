import json
import gzip

def load_json_gz(file):
    with gzip.open(file, 'rt', encoding='utf-8') as gz_file:
        anno = json.load(gz_file)
    return anno
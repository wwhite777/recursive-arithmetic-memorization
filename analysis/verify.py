"""Verify every released repository file and all imported evidence payloads."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    count=0
    for line in (BASE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);p=(BASE/name).resolve();p.relative_to(BASE);assert sha(p)==expected,name;count+=1
    for index in ['data/datasets.json','records/prediction_index.json']:
        for v in json.loads((BASE/index).read_text()):assert sha(BASE/v['path'])==v['sha256'];count+=1
    assert count>0
    print(f'PASS: {count} repository and evidence file hashes.')
if __name__=='__main__':main()

"""Verify every released repository file and all imported evidence payloads."""
import gzip,hashlib,json,re
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def verify_gpu_inventory(records):
    inventory_path=records/'gpu_inventory_20261102.json'
    inventory=json.loads(inventory_path.read_text())
    devices=inventory['gpus'];allowed={v['uuid'] for v in devices}
    assert len(devices)==len(allowed)==4 and all(v['name']=='NVIDIA L40S' for v in devices),'expected four NVIDIA L40S devices'
    assert all(re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',v) for v in allowed),'invalid inventory UUID'
    counts={'plain':Counter(),'gzip':Counter()}
    for path in sorted(records.rglob('*')):
        if not path.is_file() or path==inventory_path:continue
        kind='gzip' if path.suffix=='.gz' else 'plain'
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rb') as stream:
            for line in stream:
                for match in re.finditer(rb'GPU-[0-9a-fA-F]{8}-[A-Za-z0-9-]+',line):
                    value=match[0].decode('ascii')
                    assert value in allowed,f'unknown GPU UUID in {path.relative_to(records)}: {value}'
                    counts[kind][value]+=1
    combined={v:counts['plain'][v]+counts['gzip'][v] for v in sorted(allowed)}
    assert combined==inventory['released_record_uuid_occurrences'],'released UUID occurrence counts differ'
    for kind in counts:
        assert {v:counts[kind][v] for v in sorted(allowed)}==inventory['released_record_uuid_occurrences_'+kind],f'released {kind} UUID occurrence counts differ'
    count=sum(combined.values())
    assert count>0,'no recorded GPU UUIDs outside the inventory'
    return count
def main():
    count=0
    for line in (BASE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);p=(BASE/name).resolve();p.relative_to(BASE);assert sha(p)==expected,name;count+=1
    for index in ['data/datasets.json','records/prediction_index.json']:
        for v in json.loads((BASE/index).read_text()):assert sha(BASE/v['path'])==v['sha256'];count+=1
    assert count>0
    gpu_count=verify_gpu_inventory(BASE/'records')
    print(f'PASS: {count} repository and evidence file hashes; {gpu_count} released-record GPU UUID occurrences (inventory excluded; gzip decompressed) belong to four NVIDIA L40S devices.')
if __name__=='__main__':main()

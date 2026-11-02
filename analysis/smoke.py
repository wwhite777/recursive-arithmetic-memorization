"""Small, CPU-only generator and arithmetic check, with no model dependency."""
import hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'src'))
from reasoning.arithmetic import evaluate_rpn,group_split

def main():
    count=0;files=0
    for v in json.loads((BASE/'data/datasets.json').read_text()):
        if v['storage']=='repository':
            p=BASE/v['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];files+=1
    with (BASE/'data/id_dev.jsonl').open() as f:
        for _,line in zip(range(8),f):
            r=json.loads(line);ops=[v for v,k in zip(r['target_operator_ids'],r['target_mask']) if k]
            assert evaluate_rpn(r['structure'],r['operands'],ops)==r['result']
            assert group_split(r['operands'])=='id_dev';count+=1
    assert count==8 and files==4
    print(f'PASS: {files} dataset hashes and {count} canonical arithmetic rows; no model required.')
if __name__=='__main__':main()

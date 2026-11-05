"""Import archive payloads locally or download them with SHA256 verification."""
import argparse,gzip,hashlib,json,shutil,urllib.request,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
RECORD_FILES_URL='https://zenodo.org/records/23880859/files'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def safe(base,name):
    p=(base/name).resolve();p.relative_to(base.resolve());return p

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--archive-dir',type=Path);g.add_argument('--base-url');g.add_argument('--online',action='store_true',help='Fetch inherited payloads from the published version 1.0.1 record; supply the extension parts locally or a complete new-version --base-url.');g.add_argument('--parts-dir',type=Path);a=p.parse_args()
    cache=BASE/'data/downloads';cache.mkdir(parents=True,exist_ok=True)
    for v in json.loads((BASE/'data/archive_manifest.json').read_text()):
        dest=cache/v['file']
        if a.archive_dir:
            src=a.archive_dir/v['file'];assert digest(src)==v['sha256'],v['file']
        else:
            src=dest;partial=dest.with_suffix(dest.suffix+'.partial')
            with partial.open('wb') as out:
                for part in v['parts']:
                    h=hashlib.sha256();size=0
                    if a.parts_dir:
                        response=(a.parts_dir/part['file']).open('rb')
                    else:
                        response=urllib.request.urlopen((a.base_url or RECORD_FILES_URL).rstrip('/')+'/'+part['file']+'?download=1')
                    with response:
                        for block in iter(lambda:response.read(1<<20),b''):
                            h.update(block);size+=len(block);out.write(block)
                    assert size==part['bytes'] and h.hexdigest()==part['sha256'],part['file']
            assert digest(partial)==v['sha256'],v['file'];partial.replace(dest)
        assert src.stat().st_size==v['bytes'],v['file']
        if v['kind']=='dataset':
            target=safe(BASE,v['target']);target.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(src,'rb') as f,target.open('wb') as out:shutil.copyfileobj(f,out,1<<20)
            assert digest(target)==v['uncompressed_sha256'],v['target']
        elif v['kind']=='predictions':
            with zipfile.ZipFile(src) as archive:
                for name in archive.namelist():
                    target=safe(BASE,name);target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(name) as f,target.open('wb') as out:shutil.copyfileobj(f,out,1<<20)
        else:raise ValueError(v['kind'])
    for v in json.loads((BASE/'data/datasets.json').read_text()):assert digest(BASE/v['path'])==v['sha256'],v['path']
    for v in json.loads((BASE/'records/prediction_index.json').read_text()):assert digest(BASE/v['path'])==v['sha256'],v['path']
    print('PASS: archive, decompressed dataset and every prediction SHA256 verified.')
if __name__=='__main__':main()

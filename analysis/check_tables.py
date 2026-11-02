"""Exact, cell-by-cell comparison; every expected and actual CSV must be present."""
import argparse,csv,json
from pathlib import Path

def compare(expected,actual):
    wanted={p.name for p in expected.glob('*.csv')};found={p.name for p in actual.glob('*.csv')}
    assert wanted and wanted==found,('table inventory',sorted(wanted-found),sorted(found-wanted))
    differences=[];cells=0
    for name in sorted(wanted):
        with (expected/name).open(newline='') as f:a=list(csv.reader(f))
        with (actual/name).open(newline='') as f:b=list(csv.reader(f))
        if len(a)!=len(b):differences.append(dict(table=name,expected_rows=len(a),actual_rows=len(b)))
        for i,(x,y) in enumerate(zip(a,b)):
            if len(x)!=len(y):differences.append(dict(table=name,row=i,expected_columns=len(x),actual_columns=len(y)))
            for j,(v,w) in enumerate(zip(x,y)):
                cells+=1
                if v!=w:differences.append(dict(table=name,row=i,column=j,expected=v,actual=w))
    result=dict(tables=len(wanted),cells_including_headers=cells,differences=differences)
    print(json.dumps(result,indent=2));assert not differences,'table cells differ'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('expected',type=Path);p.add_argument('actual',type=Path);a=p.parse_args();compare(a.expected,a.actual)

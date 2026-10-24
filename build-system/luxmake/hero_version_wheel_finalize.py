"""Finalize ML HERO wheel metadata from the central header without editing wheel.py.
Usage: python hero_version_wheel_finalize.py path/to/pyluxcore-*.whl path/to/mlhero_version.h
Writes a separate hero_verified/ directory; does not overwrite the original wheel.
"""
import argparse, base64, csv, hashlib, io, json, os, re, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

def parse_header(path):
    s = Path(path).read_text(encoding='utf-8')
    d = {}
    for field, macro in [('build','ML_HERO_BUILD'),('phase','ML_HERO_TEST_PHASE'),('feature','ML_HERO_FEATURE')]:
        m = re.search(r'^\s*#\s*define\s+'+macro+r'\s+"([^"]*)"',s,re.M)
        if not m: raise ValueError('Missing central macro '+macro)
        d[field] = m.group(1)
    return d

def finalize(wheel, header):
    central = parse_header(header)
    dest_dir = wheel.parent / 'hero_verified'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / wheel.name
    with zipfile.ZipFile(wheel,'r') as zin:
        infos=zin.infolist()
        names=[i.filename for i in infos]
        records=[n for n in names if n.endswith('.dist-info/RECORD')]
        if len(records)!=1 or 'pyluxcore/ml_hero_build.json' not in names:
            raise ValueError('Expected exactly one wheel RECORD and one HERO metadata JSON')
        record=records[0]
        meta=json.loads(zin.read('pyluxcore/ml_hero_build.json'))
        meta.update(central)
        meta['version_source']='include/slg/engines/mlhero_version.h'
        meta['metadata_finalized_utc']=datetime.now(timezone.utc).isoformat(timespec='seconds')
        replacement=(json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')
        # Correct wheel RECORD hashes/sizes for every member. Python packaging specification.
        with zipfile.ZipFile(dest,'w') as zout:
            entries=[]
            for info in infos:
                if info.filename==record: continue
                content=replacement if info.filename=='pyluxcore/ml_hero_build.json' else zin.read(info.filename)
                zout.writestr(info,content)
                digest=base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode('ascii')
                entries.append((info.filename,'sha256='+digest,str(len(content))))
            entries.append((record,'',''))
            f=io.StringIO(newline='');csv.writer(f,lineterminator='\n').writerows(entries)
            zout.writestr(zin.getinfo(record),f.getvalue().encode('utf-8'))
    print('New wheel:',dest)
    print('Central identity:',central)
    print('WARNING: identity metadata is NOT proof of compiled binary version.')
    print('After installation, confirm pyluxcore.MLBuildInfo() matches this metadata.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel',type=Path);parser.add_argument('header',type=Path)
    args=parser.parse_args();finalize(args.wheel,args.header)

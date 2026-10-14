"""Download only official pinned assets on a persistent CPU server volume."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import time

REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
REPO = 'IDEA-Research/grounding-dino-tiny'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def fetch(url, path, expected=None):
    if path.exists() and (expected is None or sha(path) == expected):
        return
    partial = path.with_name(path.name+'.partial')
    for attempt in range(3):
        try:
            print('Downloading', path.name, flush=True)
            with urlopen(url, timeout=120) as response, partial.open('wb') as out:
                while chunk := response.read(1024**2):
                    out.write(chunk)
            if expected and sha(partial) != expected:
                raise ValueError('Published LFS checksum mismatch: '+path.name)
            partial.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/volume/wzhang/mesh-research-workspace/runtime/vehicle-models'),
        help='destination directory for all model files',
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    with urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=90) as response:
        meta = json.load(response)
    if meta['sha'] != REVISION:
        raise ValueError('Model revision mismatch')
    dest = root / 'grounding-dino-tiny'
    dest.mkdir(exist_ok=True)
    records = []
    for item in meta['siblings']:
        name = item['rfilename']
        if '/' in name or name.startswith('.') or not name.endswith(('.json','.txt','.safetensors')):
            continue
        path = dest / name
        expected = (item.get('lfs') or {}).get('sha256')
        fetch(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', path, expected)
        records.append(dict(name=name, sha256=sha(path), bytes=path.stat().st_size, publishedSha256=expected))
    sam = root / 'sam2.1_hiera_tiny.pt'
    url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
    fetch(url, sam, '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69')
    provenance = dict(groundingDino=dict(repository=REPO, revision=REVISION, files=records),
                      sam2=dict(url=url, sha256=sha(sam), bytes=sam.stat().st_size))
    (root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Official weights ready', flush=True)


if __name__ == '__main__':
    main()

"""Export actual draw.io pages with draw.io Desktop (PNG + SVG)."""
import json
import os
from pathlib import Path
import subprocess
import sys

DEST = Path(__file__).resolve().parent
TOOLS = Path(os.environ.get('DRAWIO_TOOLS_ROOT', '/tmp/hytrol-drawio-tools'))
env = os.environ.copy()
env['PATH'] = str(TOOLS / 'xserver/usr/bin') + ':' + env['PATH']
names = sys.argv[1:] or ['training', 'inference']
if any(name not in ['training', 'inference'] for name in names):
    raise SystemExit('Usage: export_figures.py [training] [inference]')
for name in names:
    for fmt in ['png', 'svg']:
        output = DEST / f'{name}.{fmt}'
        command = [str(TOOLS / 'xserver/usr/bin/xvfb-run'), '-a', str(TOOLS / 'app/opt/drawio/drawio'),
            '--no-sandbox', '--disable-gpu', '--user-data-dir=/tmp/synergy-amodal-drawio-profile',
            '--export', '--format', fmt, '--border', '24', '--scale', '1.5',
            '--output', str(output), str(DEST / f'{name}.drawio')]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        if result.returncode or not output.exists() or output.stat().st_size < 1000:
            raise RuntimeError(result.stdout + result.stderr)
        print(output.name, flush=True)
records = [{'file': f'{name}.{fmt}', 'bytes': (DEST / f'{name}.{fmt}').stat().st_size}
    for name in ['training', 'inference'] for fmt in ['png', 'svg'] if (DEST / f'{name}.{fmt}').exists()]
(DEST / 'export.json').write_text(json.dumps({'exporter': 'draw.io Desktop', 'scale': 1.5,
    'files': records}, indent=2)+'\n')

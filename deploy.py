#!/usr/bin/env python3
"""CostPilot Demo deployment script (server-side).

Steps: backup -> deploy -> verify md5 -> restart -> verify HTTP.
Idempotent: safe to re-run; overwrites backups on each deploy.
"""
import datetime as _dt
import hashlib
import shutil
import subprocess
import time
from pathlib import Path

APP = Path('/opt/costpilot-demo')
BUNDLE = Path('/tmp/costpilot-deploy-bundle')


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def log(msg: str) -> None:
    ts = _dt.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# 1. pre-deploy snapshot
log('=== pre-deploy snapshot ===')
r = run(['pgrep', '-af', 'uvicorn.*demo.app'])
print(r.stdout or '(no uvicorn for demo.app running)')

# 2. backup existing files
ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
log(f'=== backup (suffix .{ts}) ===')
for rel in ['demo/app.py', 'demo/static/index.html']:
    src = APP / rel
    bak = src.with_suffix(src.suffix + f'.bak-{ts}')
    if src.exists():
        shutil.copy2(src, bak)
        log(f'  {rel} -> {bak.name}')

# 3. deploy new files
log('=== deploy ===')
for rel in ['demo/app.py', 'demo/static/index.html']:
    src = BUNDLE / rel
    dst = APP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log(f'  {rel} copied')

# 4. verify md5 (local bundle vs deployed)
log('=== md5 verify ===')
all_ok = True
for rel in ['demo/app.py', 'demo/static/index.html']:
    b = md5(BUNDLE / rel)
    d = md5(APP / rel)
    ok = b == d
    all_ok = all_ok and ok
    log(f'  {rel}: bundle={b[:8]} deployed={d[:8]} {"OK" if ok else "MISMATCH"}')
if not all_ok:
    raise SystemExit('md5 mismatch, refusing to restart')

# 5. restart: prefer systemd unit; fall back to kill+spawn
log('=== restart ===')
r = run(['systemctl', 'list-units', '--type=service', '--no-legend'])
svcs = [l.split()[0] for l in r.stdout.splitlines() if 'costpilot' in l.lower()]
restarted = False
if svcs:
    for s in svcs:
        log(f'  systemctl restart {s}')
        rr = run(['systemctl', 'restart', s])
        if rr.returncode == 0:
            restarted = True
            log(f'  {s} restarted OK')
            break
        log(f'  {s} failed: {rr.stderr.strip()}')

if not restarted:
    log('  no systemd unit; killing existing uvicorn and respawning')
    run(['pkill', '-f', 'uvicorn.*demo.app'])
    time.sleep(2)
    # try common bind setups; pick the first that succeeds
    for host, port in [('127.0.0.1', 8000), ('127.0.0.1', 8001), ('0.0.0.0', 8000)]:
        venv = APP / '.venv' / 'bin' / 'python'
        if not venv.exists():
            venv = Path('/usr/bin/python3')
        cmd = [str(venv), '-m', 'uvicorn', 'demo.app:app',
               '--host', host, '--port', str(port), '--no-access-log']
        log(f'  spawning: {" ".join(cmd)}')
        with open('/var/log/costpilot-demo.log', 'ab') as logf:
            subprocess.Popen(cmd, cwd=str(APP), stdout=logf, stderr=logf, start_new_session=True)
        time.sleep(3)
        rr = run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', f'http://{host}:{port}/'])
        if rr.stdout.strip() == '200':
            log(f'  spawned on {host}:{port}')
            restarted = True
            break
        log(f'  {host}:{port} not responding (got {rr.stdout.strip()}); trying next')

if not restarted:
    raise SystemExit('failed to restart the service; manual intervention needed')

# 6. verify public HTTP
time.sleep(2)
log('=== public HTTP verify ===')
r = run(['curl', '-sk', '--resolve', 'costpilot.owain32380.cn:443:127.0.0.1',
         '-o', '/dev/null', '-w', 'HTTP %{http_code} | total %{time_total}s',
         'https://costpilot.owain32380.cn/'])
print(r.stdout)

# 7. verify the served HTML actually contains the new pricing-basis markup
r = run(['curl', '-sk', '--resolve', 'costpilot.owain32380.cn:443:127.0.0.1',
         'https://costpilot.owain32380.cn/'])
has_pb = 'id="pricingBasis"' in r.stdout
has_caret_fix = 'model=<span class="hl-s">' in r.stdout  # new caret-anchor
log(f'  pricingBasis markup present: {has_pb}')
log(f'  caret-anchor markup present: {has_caret_fix}')
if not has_pb or not has_caret_fix:
    raise SystemExit('public page missing expected new markup')

log('=== DONE: deployment verified ===')
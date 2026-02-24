import os
import random
import re
from datetime import datetime


def load_samples(raw_dir):
    samples = []
    for fname in ('malicious_local.log', 'malicious_remote.log'):
        path = os.path.join(raw_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [l.strip() for l in f if l.strip()]
                samples.extend(lines)
    if not samples:
        # fallback to a few hardcoded templates if originals missing
        samples = [
            'type=SYSCALL msg=audit(1769928916.603:5044): pid=7573 uid=0 auid=1000 ses=5 subj=unconfined msg=\'comm="sudo" exe="/usr/bin/sudo" key="process_exec"\' UID="root" AUID="heroic"',
            'type=EXECVE msg=audit(1769928916.627:5049): argc=3 a0="tail" a1="-f" a2="/var/log/audit/audit.log"',
        ]
    return samples


_TS_RE = re.compile(r'audit\((?P<sec>\d+)\.(?P<msec>\d+):(?P<id>\d+)\)')
_PID_RE = re.compile(r'pid=(?P<pid>\d+)')
_NUM_RE = re.compile(r'\b(\d{2,10})\b')
_COMM_RE = re.compile(r'comm=\"(?P<comm>[^\"]+)\"')
_EXE_RE = re.compile(r'exe=\"(?P<exe>[^\"]+)\"')
_CWD_RE = re.compile(r'cwd=\"(?P<cwd>[^\"]+)\"')


def randomize_line(template, seq, base_ts):
    line = template

    # replace audit timestamp sec.msec:id
    m = _TS_RE.search(line)
    if m:
        sec = base_ts + (seq // 1000)
        msec = random.randint(0, 999)
        msgid = random.randint(4000, 9999)
        line = _TS_RE.sub(f'audit({sec}.{msec:03d}:{msgid})', line, count=1)

    # change pid
    line = _PID_RE.sub(lambda mo: f'pid={random.randint(1000,9999)}', line)

    # random small numeric substitutions
    def num_sub(mo):
        n = int(mo.group(1))
        # keep small numbers stable, randomize larger
        if n < 100:
            return str(n)
        return str(random.randint(max(100, n - 500), n + 500))

    line = _NUM_RE.sub(num_sub, line)

    # randomize comm
    if _COMM_RE.search(line):
        comms = ['sudo', 'tail', 'sh', 'netstat', 'sed', 'sort', 'last', 'df', 'python', 'ss']
        line = _COMM_RE.sub(lambda mo: f'comm="{random.choice(comms)}"', line)

    # randomize exe
    if _EXE_RE.search(line):
        exes = ['/usr/bin/sudo', '/bin/sh', '/usr/bin/tail', '/usr/bin/netstat', '/usr/bin/sed', '/usr/bin/sort', '/usr/bin/last', '/usr/bin/df', '/usr/bin/python3']
        line = _EXE_RE.sub(lambda mo: f'exe="{random.choice(exes)}"', line)

    # randomize cwd
    if _CWD_RE.search(line):
        cwds = ['/home/heroic', '/var/ossec', '/root', '/tmp', '/usr/local/bin']
        line = _CWD_RE.sub(lambda mo: f'cwd="{random.choice(cwds)}"', line)

    return line


def generate_file(path, count, samples, start_ts):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(1, count + 1):
            tmpl = random.choice(samples)
            line = randomize_line(tmpl, i, start_ts)
            f.write(line + '\n')


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(repo_root, 'hids_dataset', 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    samples = load_samples(raw_dir)

    files = [
        ('malicious_local_01.log',),
        ('malicious_local_02.log',),
        ('malicious_local_03.log',),
        ('malicious_remote_01.log',),
        ('malicious_remote_02.log',),
        ('malicious_remote_03.log',),
        ('benign_01.log',),
        ('benign_02.log',),
        ('benign_03.log',),
    ]

    count = 1000
    base_ts = 1769928911  # base similar to provided samples

    for (name,) in files:
        path = os.path.join(raw_dir, name)
        print(f'Generating {path} ({count} lines)')
        generate_file(path, count, samples, base_ts)

    print('Done generating randomized logs.')


if __name__ == '__main__':
    main()

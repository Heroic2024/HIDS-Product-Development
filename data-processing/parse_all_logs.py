import os
import re
import json
import glob


def parse_audit_log(log_path):
    """Parse a single audit log file and return list of events."""
    events = []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or "type=EXECVE" not in line:
                continue
            
            # Extract timestamp
            ts_match = re.search(r"audit\((\d+\.\d+):", line)
            timestamp = float(ts_match.group(1)) if ts_match else None
            
            # Extract arguments (command-line args)
            args = re.findall(r'a\d+="([^"]+)"', line)
            command = " ".join(args) if args else ""
            
            # Build event object
            event = {
                "timestamp": timestamp,
                "event_type": "process_exec",
                "process": args[0] if args else None,
                "command": command,
                "source": "auditd"
            }
            
            events.append(event)
    
    return events


def get_category_from_filename(filename):
    """Determine category (benign, malicious_local, malicious_remote) from filename."""
    if 'benign' in filename:
        return 'benign'
    elif 'malicious_remote' in filename:
        return 'malicious_remote'
    elif 'malicious_local' in filename:
        return 'malicious_local'
    return 'unknown'


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(repo_root, 'hids_dataset', 'raw')
    parsed_dir = os.path.join(repo_root, 'hids_dataset', 'parsed')
    
    os.makedirs(parsed_dir, exist_ok=True)
    
    # Find all log files
    log_files = sorted(glob.glob(os.path.join(raw_dir, '*.log')))
    
    if not log_files:
        print(f'No log files found in {raw_dir}')
        return
    
    # Parse each log file
    for log_path in log_files:
        basename = os.path.basename(log_path)
        stem = os.path.splitext(basename)[0]  # remove .log
        
        print(f'Parsing {basename}...')
        events = parse_audit_log(log_path)
        
        if not events:
            print(f'  [!] No EXECVE events found in {basename}')
            continue
        
        # Output file
        out_filename = f'parsed_{stem}.json'
        out_path = os.path.join(parsed_dir, out_filename)
        
        with open(out_path, 'w', encoding='utf-8') as out:
            json.dump(events, out, indent=2)
        
        print(f'  [+] Parsed {len(events)} events -> {out_filename}')
    
    print('\nDone parsing all logs.')


if __name__ == '__main__':
    main()

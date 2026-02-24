import re
import json

AUDIT_LOG = r"C:\Users\Aryan Mhatre\HIDS-Product-Development\hids_dataset\raw\malicious_remote.log"
OUTPUT = "parsed_malicious_remote.json"

events = []

with open(AUDIT_LOG, "r") as f:
    for line in f:
        if "type=EXECVE" not in line:
            continue

        # Timestamp
        ts_match = re.search(r"audit\((\d+\.\d+):", line)
        timestamp = float(ts_match.group(1)) if ts_match else None

        # Extract arguments
        args = re.findall(r'a\d+="([^"]+)"', line)
        command = " ".join(args)

        event = {
            "timestamp": timestamp,
            "event_type": "process_exec",
            "process": args[0] if args else None,
            "command": command,
            "source": "auditd"
        }

        events.append(event)

with open(OUTPUT, "w") as out:
    json.dump(events, out, indent=2)

print(f"[+] Parsed {len(events)} audit events")

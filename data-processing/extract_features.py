import os
import json
import glob
import re
from collections import Counter, defaultdict
import math


def extract_features_from_events(events):
    """Extract behavioral and statistical features from event list."""
    if not events:
        return None
    
    features = {}
    
    # Temporal features
    if len(events) > 1:
        timestamps = [e.get('timestamp', 0) for e in events]
        timestamps = [t for t in timestamps if t]  # filter None
        if timestamps:
            timestamps.sort()
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            features['event_count'] = len(events)
            features['avg_time_delta'] = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            features['max_time_delta'] = max(time_diffs) if time_diffs else 0
            features['min_time_delta'] = min(time_diffs) if time_diffs else 0
            
            # Standard deviation
            if time_diffs:
                mean = features['avg_time_delta']
                variance = sum((x - mean) ** 2 for x in time_diffs) / len(time_diffs)
                features['std_time_delta'] = math.sqrt(variance)
            else:
                features['std_time_delta'] = 0
        else:
            features['event_count'] = len(events)
            features['avg_time_delta'] = 0
            features['max_time_delta'] = 0
            features['min_time_delta'] = 0
            features['std_time_delta'] = 0
    else:
        features['event_count'] = len(events)
        features['avg_time_delta'] = 0
        features['max_time_delta'] = 0
        features['min_time_delta'] = 0
        features['std_time_delta'] = 0
    
    # Process features
    processes = [e.get('process') for e in events if e.get('process')]
    commands = [e.get('command') for e in events if e.get('command')]
    
    features['unique_processes'] = len(set(processes))
    features['unique_commands'] = len(set(commands))
    features['total_processes'] = len(processes)
    features['total_commands'] = len(commands)
    
    # Process frequency (top 5)
    if processes:
        proc_freq = Counter(processes)
        features['top_process_freq'] = proc_freq.most_common(1)[0][1] if proc_freq else 0
        features['process_entropy'] = calculate_entropy(processes)
    else:
        features['top_process_freq'] = 0
        features['process_entropy'] = 0.0
    
    # Command features
    if commands:
        cmd_lengths = [len(c) for c in commands]
        features['avg_command_length'] = sum(cmd_lengths) / len(cmd_lengths) if cmd_lengths else 0
        features['max_command_length'] = max(cmd_lengths) if cmd_lengths else 0
        features['min_command_length'] = min(cmd_lengths) if cmd_lengths else 0
        features['cmd_entropy'] = calculate_entropy(commands)
    else:
        features['avg_command_length'] = 0
        features['max_command_length'] = 0
        features['min_command_length'] = 0
        features['cmd_entropy'] = 0.0
    
    # Suspicious pattern detection
    features['suspicious_patterns'] = detect_suspicious_patterns(commands)
    features['contains_sudo'] = sum(1 for c in commands if 'sudo' in c.lower())
    features['contains_pipe'] = sum(1 for c in commands if '|' in c)
    features['contains_redirect'] = sum(1 for c in commands if any(x in c for x in ['>', '<', '>>', '&']))
    
    return features


def calculate_entropy(items):
    """Calculate Shannon entropy for a list of items."""
    if not items:
        return 0.0
    counter = Counter(items)
    total = len(items)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def detect_suspicious_patterns(commands):
    """Count suspicious command patterns."""
    suspicious = 0
    patterns = [
        r'nc\s+-l',  # netcat listener
        r'rm\s+-rf',  # recursive delete
        r'/dev/zero',  # bit flip
        r'chmod\s+777',  # wide permissions
        r'eval|exec|source',  # code execution
        r'wget|curl|wget',  # file download
        r'base64',  # encoding/obfuscation
        r'iptables|ufw',  # firewall manipulation
    ]
    
    for cmd in commands:
        for pattern in patterns:
            if re.search(pattern, cmd, re.IGNORECASE):
                suspicious += 1
                break
    
    return suspicious


def categorize_log(filename):
    """Determine category from filename: benign, malicious_local, malicious_remote."""
    if 'benign' in filename:
        return 'benign'
    elif 'malicious_remote' in filename:
        return 'malicious_remote'
    elif 'malicious_local' in filename:
        return 'malicious_local'
    return 'unknown'


def save_csv(data, path):
    """Save list of dicts as CSV without pandas."""
    if not data:
        return
    
    keys = list(data[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        # Header
        f.write(','.join(keys) + '\n')
        # Rows
        for row in data:
            values = []
            for key in keys:
                val = row.get(key, '')
                # Handle numeric types and strings
                if isinstance(val, str):
                    val = val.replace(',', ';').replace('\n', ' ')
                    val = f'"{val}"'
                values.append(str(val))
            f.write(','.join(values) + '\n')


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    parsed_dir = os.path.join(repo_root, 'hids_dataset', 'parsed')
    features_dir = os.path.join(repo_root, 'hids_dataset', 'features')
    
    os.makedirs(features_dir, exist_ok=True)
    
    # Find all parsed JSON files
    parsed_files = sorted(glob.glob(os.path.join(parsed_dir, 'parsed_*.json')))
    
    if not parsed_files:
        print(f'No parsed JSON files found in {parsed_dir}')
        return
    
    # Collect all features and metadata
    all_features = []
    category_features = {'benign': [], 'malicious_local': [], 'malicious_remote': []}
    category_counts = defaultdict(int)
    
    print('Extracting features from parsed logs...\n')
    
    for json_path in parsed_files:
        filename = os.path.basename(json_path)
        stem = os.path.splitext(filename)[0]  # e.g., parsed_benign_01
        category = categorize_log(filename)
        
        print(f'Processing {filename}...')
        
        # Load parsed events
        with open(json_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        # Extract features
        features = extract_features_from_events(events)
        
        if features:
            features['filename'] = stem
            features['original_category'] = category
            features['label'] = 1 if category == 'benign' else 0  # 1=benign, 0=malicious
            all_features.append(features)
            category_features[category].append(features)
            category_counts[category] += 1
            print(f'  [+] Extracted {features["event_count"]} events, {features["unique_processes"]} unique processes')
    
    # Create dataset CSV
    csv_path = os.path.join(features_dir, 'dataset_features.csv')
    save_csv(all_features, csv_path)
    print(f'\n[+] Saved dataset with {len(all_features)} samples to dataset_features.csv')
    
    # Save feature summary JSON
    summary = {
        'total_samples': len(all_features),
        'categories': dict(category_counts),
        'feature_columns': list(all_features[0].keys()) if all_features else [],
        'benign_count': category_counts['benign'],
        'malicious_local_count': category_counts['malicious_local'],
        'malicious_remote_count': category_counts['malicious_remote'],
    }
    
    summary_path = os.path.join(features_dir, 'dataset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'[+] Saved dataset summary to dataset_summary.json')
    
    # Save per-category feature files
    for category in ['benign', 'malicious_local', 'malicious_remote']:
        if category_features[category]:
            cat_path = os.path.join(features_dir, f'features_{category}.csv')
            save_csv(category_features[category], cat_path)
            print(f'[+] Saved {len(category_features[category])} {category} features to features_{category}.csv')
    
    # Save combined JSON
    json_path = os.path.join(features_dir, 'dataset_features.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, indent=2)
    print(f'[+] Saved dataset features to dataset_features.json')
    
    # Statistical summary
    print('\n=== Dataset Statistics ===')
    print(f'Total samples: {len(all_features)}')
    print(f'Benign: {category_counts["benign"]}')
    print(f'Malicious (Local): {category_counts["malicious_local"]}')
    print(f'Malicious (Remote): {category_counts["malicious_remote"]}')
    if all_features:
        print(f'\nFeature columns ({len(all_features[0])}): {list(all_features[0].keys())}')
    
    print('\nDone extracting features.')


if __name__ == '__main__':
    main()

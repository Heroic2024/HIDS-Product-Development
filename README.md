# HIDS-Product-Development
This project implements an AI-driven Host-Based Intrusion Detection System (HIDS) for Linux environments, focused on behavioral detection using system telemetry rather than signature-based rules.
This project implements an AI-driven Host-Based Intrusion Detection System (HIDS) designed for Linux environments, focusing on behavioral detection rather than signature-based methods. The system collects kernel-level telemetry using auditd and analyzes system behavior to detect malicious activities such as persistence attempts, privilege escalation patterns, living-off-the-land attacks, ransomware-like file activity, and remote intrusion attempts.

A custom dataset was created by automating benign workloads and simulating both local and remote attack scenarios in an isolated virtual lab. Local attack simulations included reconnaissance, misuse of legitimate binaries (LOLBins), cron-based persistence, and mass file creation, while remote attacks were generated from a Kali Linux machine to emulate network scanning and SSH intrusion attempts. All activities were logged using auditd, ensuring realistic and reproducible telemetry.

The raw audit logs were parsed, normalized, and segmented into time-based windows, from which behavioral features were extracted. Machine learning models were trained on a high-performance system and deployed for lightweight inference, enabling real-time detection with low overhead. The project emphasizes explainability, correlating alerts with event timelines to clearly justify why an activity was classified as malicious.

This system demonstrates how AI-augmented behavioral analysis can improve host-level intrusion detection, reduce false positives, and provide practical detection capabilities aligned with modern SOC and EDR workflows.

## Data Preprocessing

The dataset preparation pipeline includes the following steps:

1. **Synthetic Log Generation**: Generated ~1000-line audit log files for three categories (benign, malicious_local, malicious_remote) with randomized timestamps, process IDs, commands, and file paths to ensure diversity for model training.

2. **Log Parsing**: Converted raw auditd logs into structured JSON format, extracting EXECVE events with timestamps, process names, and command arguments. Resulted in 1,588 total events across 12 parsed log files.

3. **Feature Extraction**: Computed 19 behavioral features from event sequences:
   - **Temporal features**: event count, time deltas (avg, max, min, std)
   - **Process features**: unique/total processes, process frequency, entropy
   - **Command features**: unique/total commands, command length metrics, entropy
   - **Behavioral indicators**: suspicious patterns, sudo usage, pipes, redirects

4. **Dataset Creation**: Generated train/test CSV datasets with 80-20 split (10,009 training samples, 10,003 test samples), standardized using StandardScaler for scikit-learn compatibility.

5. **Data Augmentation**: Expanded datasets with synthetic samples maintaining class distribution (~33% benign, ~67% malicious) to reach 20,000+ total samples ready for model training.

**Output**: Standardized train.csv and test.csv with 19 features + binary label (1=benign, 0=malicious).

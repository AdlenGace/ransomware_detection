import os
import pandas as pd
import pyshark

def collect_sysmon_logs(log_path):
    """
    Collect logs from Sysmon (or any other monitoring tool).
    Args:
        log_path (str): Path to the log file or directory.
    Returns:
        pd.DataFrame: Collected logs in a pandas DataFrame.
    """
    logs = []
    for filename in os.listdir(log_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(log_path, filename)
            df = pd.read_csv(file_path)
            logs.append(df)
    
    if logs:
        return pd.concat(logs, ignore_index=True)
    else:
        print("No Sysmon CSV logs found.")
        return pd.DataFrame()


def collect_pcap_logs(log_path):
    """
    Collect logs from .pcap files using pyshark.
    Args:
        log_path (str): Path to the .pcap file or directory.
    Returns:
        pd.DataFrame: Collected logs in a pandas DataFrame.
    """
    logs = []
    for filename in os.listdir(log_path):
        if filename.endswith('.pcap'):
            file_path = os.path.join(log_path, filename)
            print(f"Processing {file_path}...")

            # Use pyshark to read the pcap file
            capture = pyshark.FileCapture(file_path)

            for packet in capture:
                # Example of extracting packet info (can be customized further)
                try:
                    timestamp = packet.sniff_time
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                    protocol = packet.transport_layer

                    logs.append({
                        'timestamp': timestamp,
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'protocol': protocol,
                    })

                except AttributeError:
                    # In case the packet doesn't have IP or transport layer info
                    continue

    print(f"Collected {len(logs)} packets from .pcap file.")
    return pd.DataFrame(logs)


def preprocess_logs(logs):
    """
    Preprocess the logs by handling missing data and converting timestamps.
    Args:
        logs (pd.DataFrame): Raw logs.
    Returns:
        pd.DataFrame: Cleaned logs.
    """
    # Convert timestamp to datetime object
    if 'timestamp' in logs.columns:
        logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors='coerce')
    
    # Fill missing values if necessary
    logs.fillna(method='ffill', inplace=True)
    
    return logs


def main():
    log_path = 'data/raw_logs'  # Path to the raw logs
    print(f"Collecting logs from {log_path}...")

    # Collect .pcap logs
    pcap_logs = collect_pcap_logs(log_path)
    
    if pcap_logs.empty:
        print("No .pcap logs collected.")
        return

    # Preprocess the logs
    processed_logs = preprocess_logs(pcap_logs)

    # Save the processed logs to the processed_data directory
    processed_data_path = 'data/processed_data/processed_logs.csv'
    processed_logs.to_csv(processed_data_path, index=False)
    print(f"Processed logs saved to {processed_data_path}")


if __name__ == "__main__":
    main()
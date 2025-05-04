import os
import pandas as pd

def extract_features(logs):
    """
    Extract relevant features from the collected logs.
    Args:
        logs (pd.DataFrame): Processed logs DataFrame
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    # Example features: packet counts, IP addresses, protocols, etc.
    
    # Count packets per source IP address
    packet_counts = logs['src_ip'].value_counts().reset_index()
    packet_counts.columns = ['ip_address', 'packet_count']
    
    # Count packets per destination IP address
    dst_packet_counts = logs['dst_ip'].value_counts().reset_index()
    dst_packet_counts.columns = ['ip_address', 'packet_count']
    
    # Combine source and destination IP counts
    ip_counts = pd.concat([packet_counts, dst_packet_counts], ignore_index=True)
    ip_counts = ip_counts.groupby('ip_address').sum().reset_index()

    # Calculate protocol counts (does not need merging with ip_counts)
    protocol_counts = logs['protocol'].value_counts().reset_index()
    protocol_counts.columns = ['protocol', 'count']
    
    # Combine features: ip_counts and protocol_counts
    features = pd.merge(ip_counts, protocol_counts, how='outer', left_on='ip_address', right_on='protocol')
    
    return features

def main():
    """
    Main function to load processed logs, extract features, and save the features.
    """
    processed_data_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/processed_data/processed_logs.csv'  # Adjust path as needed
    output_path = '/home/adlen/Desktop/Projects/ransomware_detections_project/data/processed_data/extracted_features.csv'  # Path to save features
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(processed_data_path):
        # Load the processed logs
        logs = pd.read_csv(processed_data_path)
        
        # Extract features
        print("Extracting features from logs...")
        features = extract_features(logs)
        
        # Save features to a CSV file
        features.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
    else:
        print(f"Processed data not found at {processed_data_path}")

if __name__ == "__main__":
    main()

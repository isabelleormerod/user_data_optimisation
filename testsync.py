import json
import pandas as pd
import numpy as np

def sync_tracking_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Extract Hand Tracking (The "Video" Clock)
    # This uses Unity Simulation Time (starts at ~0)
    if 'handTracking' in data and data['handTracking']['frames']:
        hand_frames = data['handTracking']['frames']
        df_hands = pd.DataFrame([{
            'sim_time': f['timestamp'],
            'frame_idx': f['frame']
        } for f in hand_frames])
    else:
        print("Error: No Hand Tracking data found.")
        return None, None, None

    # 2. Extract Pen Tracking (The "Real World" Clock)
    # This uses Unix Milliseconds (e.g., 1774622005228)
    if 'penTracking' in data and data['penTracking']['frames']:
        pen_frames = data['penTracking']['frames']
        df_pen = pd.DataFrame([{
            'real_time_ms': f['timestamp'],
            'pos_x': f['position']['x'],
            'pos_y': f['position']['y'],
            'pos_z': f['position']['z']
        } for f in pen_frames])
    else:
        print("Error: No Pen Tracking data found.")
        return df_hands, None, None

    # --- THE FIX: Calculate Warping Ratio ---
    # .iloc[-1] is the last value, .iloc is the first value
    real_duration_ms = df_pen['real_time_ms'].iloc[-1] - df_pen['real_time_ms'].iloc
    sim_duration_s = df_hands['sim_time'].iloc[-1] - df_hands['sim_time'].iloc
    
    # Calculate how many real seconds occurred per simulation second
    # If ratio is 2.0, it means 2 seconds of real life were squeezed into 1 second of video
    ratio = (real_duration_ms / 1000.0) / sim_duration_s
    
    print(f"\n--- Sync Report: {json_path} ---")
    print(f"Detected Lag Factor: {ratio:.4f}x (Ratio of Real Time to Video Time)")
    
    # 3. Create the Synced Timeline for the Pen
    start_real = df_pen['real_time_ms'].iloc
    start_sim = df_hands['sim_time'].iloc

    # Warp the pen timestamps to match the hands/video timeline
    df_pen['synced_video_time'] = ((df_pen['real_time_ms'] - start_real) / 1000.0) / ratio + start_sim

    # 4. Handle Body Tracking (MediaPipe)
    df_body = None
    if 'bodyTracking' in data and data['bodyTracking']['frames']:
        body_list = []
        for f in data['bodyTracking']['frames']:
            if f['poses']: # Only grab frames with a detected person
                # Storing just the timestamp and the first pose for this example
                body_list.append({
                    'real_time_ms': f['timestamp'],
                    'nose_x': f['poses']['landmarks']['x']
                })
        
        if body_list:
            df_body = pd.DataFrame(body_list)
            # Sync body using the same ratio as the pen
            df_body['synced_video_time'] = ((df_body['real_time_ms'] - start_real) / 1000.0) / ratio + start_sim
            print(f"Body Tracking Synced: {len(df_body)} frames.")

    return df_hands, df_pen, df_body

# --- EXECUTION ---
file_to_process = 'P002_Long_Small_Front_weighted_A180.json'
df_hands, df_pen, df_body = sync_tracking_data(file_to_process)

if df_pen is not None:
    print("\nSample of Synced Pen Data:")
    print(df_pen[['real_time_ms', 'synced_video_time']].head())
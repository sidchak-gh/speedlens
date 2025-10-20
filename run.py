"""
Wrapper script to run the vehicle tracking with proper PyTorch configuration
"""
import os
import sys
import warnings

# Add ByteTrack to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ByteTrack'))

# Suppress PyTorch serialization warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Monkey patch torch.load to use weights_only=False
import torch
original_load = torch.load

def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

# Now import and run the main script
if __name__ == '__main__':
    from main import vehicle_tracker_and_counter
    
    input_video = "assets/vehicle-counting.mp4"
    output_video = "assets/vehicle-counting-result.mp4"
    
    pipeline = vehicle_tracker_and_counter(
        source_video_path=input_video,
        target_video_path=output_video,
        use_tensorrt=False
    )
    pipeline.run()

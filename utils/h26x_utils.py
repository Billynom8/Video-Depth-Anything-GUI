import subprocess
import os
import logging

# Define the FFmpeg command template and parameters
def run_h26x_encoding(input_dir, output_path, codec_mode, fps, crf, invert_vis, write_event_value):
    """
    Runs an FFmpeg subprocess to encode a PNG sequence into a high-bit-depth H.26x video.
    
    :param input_dir: Path to the folder containing frame_00000.png, etc.
    :param output_path: Full path to the output video file (e.g., ..._vis.mp4)
    :param codec_mode: One of the H.26x options (libx264, libx265, nvenc_h264, nvenc_h265)
    :param fps: Video framerate.
    :param crf: CRF value for quality control.
    :param invert_vis: Boolean flag to invert the visualization (apply max-value).
    :param write_event_value: Function to send status updates back to the GUI thread.
    """
    
    # 1. Determine Codec, Profile, and Pixel Format based on mode
    codec_map = {
        'libx264 (8-bit)': ('libx264', 'high', 'yuv420p', False),
        'libx265 (10-bit)': ('libx265', 'main10', 'yuv420p10le', True),
        'nvenc_h264 (8-bit)': ('h264_nvenc', 'high', 'yuv420p', False),
        'nvenc_h265 (10-bit)': ('hevc_nvenc', 'main10', 'yuv420p10le', True),
    }

    try:
        codec, profile, pix_fmt, is_10bit = codec_map[codec_mode]
    except KeyError:
        write_event_value('-ERROR-', f"Internal error: Unknown codec mode '{codec_mode}'")
        return

    # 2. Build the Input/Filter/Output Command Components
    
    # Input sequence uses 16-bit PNGs (which FFmpeg reads as gray16le)
    input_args = [
        '-y',                           # Overwrite output files without asking
        '-i', os.path.join(input_dir, 'frame_%05d.png'),
        '-r', str(fps),
    ]

    # V-filter for inverting visualization (Only apply if invert_vis is True)
    # The depth map is 16-bit grayscale (0-65535). Applying the curve/LUT to this.
    filter_args = []
    if invert_vis:
        # Curve 'opposit' flips the tone curve (closer=lighter -> closer=darker)
        # This is a robust way to invert visualization for 16-bit data
        # Scale to 8-bit first for visualization purposes (FFmpeg LUTs usually expect 8-bit range)
        filter_args.extend([
            '-vf', f'format=gray, lut=curve=opposit',
        ])
    else:
        # Convert 16-bit gray input to 8-bit gray for visualization without inversion
        filter_args.extend([
            '-vf', 'format=gray'
        ])
        
    # Standard output arguments
    output_args = [
        '-vcodec', codec,
        '-pix_fmt', pix_fmt,
        '-crf', str(crf),
        '-profile:v', profile,
        '-tag:v', 'hvc1' if is_10bit else 'avc1', # Tag for better compatibility (HEVC/AVC)
        output_path
    ]

    # 3. Execute FFmpeg
    command = ['ffmpeg'] + input_args + filter_args + output_args
    
    logging.info(f"FFmpeg Command: {' '.join(command)}")
    
    try:
        # Use subprocess.run to execute the command
        # Capture output only to send errors back to GUI
        subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            encoding='utf-8'
        )
        write_event_value('-STATUS_UPDATE-', f'Encoding complete: {os.path.basename(output_path)}')
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg Error for {os.path.basename(output_path)}:\n"
        error_msg += f"Command: {' '.join(command)}\n"
        error_msg += f"Return Code: {e.returncode}\n"
        error_msg += f"Stderr: {e.stderr}"
        logging.error(error_msg)
        write_event_value('-ERROR-', error_msg)
    except FileNotFoundError:
        write_event_value('-ERROR-', "FFmpeg executable not found. Please ensure FFmpeg is installed and in your system PATH.")
    except Exception as e:
        write_event_value('-ERROR-', f"An unexpected error occurred during encoding: {str(e)}")

# Add a check at the bottom to warn if FFmpeg isn't available
try:
    subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("FFmpeg is available in system PATH.")
except Exception:
    logging.warning("FFmpeg not found in system PATH. Video encoding may fail for custom modes.")
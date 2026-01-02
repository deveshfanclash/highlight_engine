import sys, os
import traceback
from utils.log_utils import logger

def frame_to_timecode(frame_idx, fps: float, format: str = "HH:MM:SS.mmm") -> str: # HH:MM:SS.mmm format
    try:
        total_seconds = frame_idx / fps
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60
        # return f"{h:02d}:{m:02d}:{s:06.3f}"
        if format == "HH:MM:SS.mmm":
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        elif format == "HH:MM:SS":
            return f"{h:02d}:{m:02d}:{int(s):02d}"
        elif format == "Seconds":
            return f"{total_seconds:.3f}"
        elif format == "HH_MM_SS":
            return f"{h:02d}_{m:02d}_{int(s):02d}"
        elif format == "HH_MM_SS_mmm":
            return f"{h:02d}_{m:02d}_{s:02.3f}"

    except Exception as e:
        print(f"[ERROR] frame_to_timecode: {e}")
        print(traceback.format_exc())
        return "00:00:00.000"
if __name__ =="__main__":
    #For testing individual functions
    sys.path.insert(0, os.getcwd())
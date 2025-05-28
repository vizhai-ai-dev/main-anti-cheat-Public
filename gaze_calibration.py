#!/usr/bin/env python3
"""
Simple gaze calibration tool to test and adjust thresholds
"""

import cv2
from gaze_tracking import get_gaze_direction

def main():
    print("Gaze Calibration Tool")
    print("=" * 40)
    print("Look in different directions and see the values:")
    print("- Look CENTER, LEFT, RIGHT, UP, DOWN")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset/recalibrate")
    print("=" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get gaze with debug info
        direction, debug_info = get_gaze_direction(frame, debug=True)
        
        # Display frame with info
        display_frame = frame.copy()
        
        # Add direction text
        color = (0, 255, 0) if direction != "unknown" else (0, 0, 255)
        cv2.putText(display_frame, f"Direction: {direction.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add debug values if available
        if 'average_gaze_ratio' in debug_info:
            h_ratio = debug_info['average_gaze_ratio']
            v_ratio = debug_info['average_vertical_ratio']
            
            cv2.putText(display_frame, f"H-Ratio: {h_ratio:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"V-Ratio: {v_ratio:.3f}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show thresholds
            cv2.putText(display_frame, "Thresholds:", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_frame, "H: 0.85 < center < 1.15", 
                       (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(display_frame, "V: 0.8 < center < 1.2", 
                       (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow('Gaze Calibration', display_frame)
        
        # Print values every 30 frames for console monitoring
        frame_count += 1
        if frame_count % 30 == 0 and 'average_gaze_ratio' in debug_info:
            h_ratio = debug_info['average_gaze_ratio']
            v_ratio = debug_info['average_vertical_ratio']
            print(f"Direction: {direction:8} | H: {h_ratio:.3f} | V: {v_ratio:.3f}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Recalibrating... Look at camera center")
            frame_count = 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
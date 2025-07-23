import cv2
import numpy as np
import serial
import time

# Set up serial communication with Arduino
arduino = serial.Serial('COM1', 9600)  # Replace 'COM1' with your Arduino's serial port
time.sleep(2)  # Wait for the serial connection to initialize

# Define the lower and upper bounds for the colors in HSV space
color_ranges = {
    'y': ((20, 100, 100), (30, 255, 255)),  # Yellow
    'g': ((40, 50, 50), (90, 255, 255)),    # Green
    'b': ((100, 150, 150), (140, 255, 255)), # Refined Blue
    'r1': ((0, 70, 50), (10, 255, 255)),    # First red range
    'r2': ((170, 70, 50), (180, 255, 255))  # Second red range
}

def detect_color_and_location(frame, lower_bound, upper_bound):
    """Detect the color within the specified bounds in the given frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    
    # Create a mask to exclude low saturation and low value
    hsv_s = hsv_frame[:, :, 1]
    hsv_v = hsv_frame[:, :, 2]
    mask = cv2.bitwise_and(mask, mask, mask=(hsv_s > 50).astype(np.uint8) & (hsv_v > 50).astype(np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Adjust size threshold as needed
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return (x, y, w, h), (center_x, center_y)
    
    return None, None

def main():
    cap = cv2.VideoCapture(0)  # Open the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        for color_code, (lower_bound, upper_bound) in color_ranges.items():
            bbox, center = detect_color_and_location(frame, lower_bound, upper_bound)
            
            if bbox and center:
                x, y, w, h = bbox
                center_x, center_y = center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # If color code is 'r1' or 'r2', label it as 'R'
                display_color = 'R' if color_code in ['r1', 'r2'] else color_code.upper()
                cv2.putText(frame, display_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

                # Send the color code and location to Arduino
                data = f"{display_color},{center_x},{center_y}\n"
                print(f"Sending data to Arduino: {data}")
                arduino.write(data.encode())
                time.sleep(0.5)  # Short delay to prevent flooding the serial port

        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    arduino.close()

if __name__ == "__main__":
    main()

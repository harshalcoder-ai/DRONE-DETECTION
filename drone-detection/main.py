import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import numpy as np
import serial
import serial.tools.list_ports


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Detection, Tracking & Neutralize")
        self.root.geometry("900x700")
        
        # Initialize YOLO model
        try:
            self.model = YOLO('epoch90.pt')
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
        
        # Detection settings
        self.detection_enabled = True
        self.confidence_threshold = 0.75
        self.detection_count = 0
        
        # Crosshair and targeting settings
        self.show_crosshair = True
        self.crosshair_color_normal = (0, 0, 255)  # Red color (BGR format)
        self.crosshair_color_target = (0, 255, 0)  # Green color when target acquired (BGR format)
        self.crosshair_thickness = 2
        self.center_square_size = 50
        
        # Target acquisition settings
        self.target_acquired = False
        self.center_tolerance = 75  # Pixels from center to consider "centered"
        
        # Arduino serial communication settings
        self.arduino_serial = None
        self.arduino_connected = False
        self.serial_port = None
        self.baud_rate = 9600
        
        # Position tracking for motor control
        self.drone_position = {"x": "center", "y": "center"}  # left/right/center, top/bottom/center
        self.position_deadzone = 100  # Pixels from center to consider as "center" position
        
        # Initialize Arduino connection
        self.setup_arduino_connection()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # 0 for default camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            self.root.destroy()
            return
        
        # Create GUI elements
        self.setup_gui()
        
        # Flag to control video loop
        self.running = True
        
        # Start video loop in separate thread
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_arduino_connection(self):
        """Initialize Arduino serial connection"""
        print("=== SETTING UP ARDUINO CONNECTION ===")
        try:
            # Auto-detect Arduino port
            ports = serial.tools.list_ports.comports()
            arduino_port = None
            
            print("Scanning for Arduino devices...")
            print("Available ports:")
            for port in ports:
                print(f"  - {port.device}: {port.description}")
                # Look for Arduino-like devices
                if ('Arduino' in port.description or 
                    'USB' in port.description or 
                    'CH340' in port.description or
                    'CP210' in port.description or
                    'FTDI' in port.description or
                    'Serial' in port.description):
                    arduino_port = port.device
                    print(f"  ✅ Found potential Arduino: {port.device}")
            
            if arduino_port:
                print(f"Attempting to connect to: {arduino_port}")
                self.arduino_serial = serial.Serial(arduino_port, self.baud_rate, timeout=1)
                self.arduino_connected = True
                self.serial_port = arduino_port
                print(f"✅ Arduino connected successfully on port: {arduino_port}")
                print("Waiting 2 seconds for Arduino to initialize...")
                time.sleep(2)  # Wait for Arduino to initialize
                print("Arduino should be ready now!")
            else:
                print("❌ No Arduino found in available ports")
                self.arduino_connected = False
                self.arduino_serial = None
                self.serial_port = None
        except Exception as e:
            print(f"❌ Error connecting to Arduino: {e}")
            self.arduino_connected = False
            self.arduino_serial = None
            self.serial_port = None
        
        print("=== ARDUINO CONNECTION SETUP COMPLETE ===")
        print(f"Connection status: {'✅ Connected' if self.arduino_connected else '❌ Disconnected'}")
    
    def send_arduino_command(self, command):
        """Send command to Arduino via serial"""
        print(f"Attempting to send command: {command}")
        print(f"Arduino connected: {self.arduino_connected}")
        print(f"Arduino serial object: {self.arduino_serial}")
        
        if self.arduino_connected and self.arduino_serial:
            try:
                self.arduino_serial.write(f"{command}\n".encode())
                print(f"✅ Successfully sent to Arduino: {command}")
                return True
            except Exception as e:
                print(f"❌ Error sending command to Arduino: {e}")
                self.arduino_connected = False
                return False
        else:
            print("❌ Arduino not connected or serial object is None")
            return False
    
    def calculate_drone_position(self, drone_center_x, drone_center_y, frame_width, frame_height):
        """Calculate drone position relative to camera center"""
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Calculate horizontal position
        if drone_center_x < center_x - self.position_deadzone:
            x_position = "left"
        elif drone_center_x > center_x + self.position_deadzone:
            x_position = "right"
        else:
            x_position = "center"
        
        # Calculate vertical position
        if drone_center_y < center_y - self.position_deadzone:
            y_position = "top"
        elif drone_center_y > center_y + self.position_deadzone:
            y_position = "bottom"
        else:
            y_position = "center"
        
        return {"x": x_position, "y": y_position}
    
    def send_motor_commands(self, position):
        """Send motor movement commands based on drone position"""
        if not self.arduino_connected:
            return
            
        # Send horizontal movement command
        if position["x"] == "left":
            self.send_arduino_command("RIGHT")
        elif position["x"] == "right":
            self.send_arduino_command("LEFT")
        
        # Send vertical movement command
        if position["y"] == "top":
            self.send_arduino_command("UP")
        elif position["y"] == "bottom":
            self.send_arduino_command("DOWN")
        
        # Send center command if drone is centered
        if position["x"] == "center" and position["y"] == "center":
            self.send_arduino_command("SHOOT")
            time.sleep(0.3)
            self.send_arduino_command("STOP")

    
    def reconnect_arduino(self):
        """Reconnect to Arduino"""
        if self.arduino_serial:
            try:
                self.arduino_serial.close()
            except:
                pass
        self.setup_arduino_connection()
        
        # Update GUI status
        arduino_status = "Connected" if self.arduino_connected else "Disconnected"
        self.arduino_status_label.configure(text=f"Arduino: {arduino_status}")
        port_info = f"Port: {self.serial_port}" if self.serial_port else "Port: None"
        self.port_label.configure(text=port_info)
    
    def test_arduino_connection(self):
        """Test Arduino connection by sending test commands"""
        print("=== TESTING ARDUINO CONNECTION ===")
        
        # Test basic commands
        test_commands = ["STATUS", "CENTER", "LEFT", "RIGHT", "UP", "DOWN", "SHOOT"]
        
        for command in test_commands:
            success = self.send_arduino_command(command)
            if not success:
                print(f"Failed to send {command} - stopping test")
                break
            time.sleep(0.5)  # Small delay between commands

        
        print("=== TEST COMPLETE ===")
        
        # Update GUI status
        arduino_status = "Connected" if self.arduino_connected else "Disconnected"
        self.arduino_status_label.configure(text=f"Arduino: {arduino_status}")
    
    def show_settings(self):
        """Show settings dialog (placeholder for future implementation)"""
        print("⚙️ Settings dialog - Coming soon!")
        print("Future settings options:")
        print("- Confidence threshold adjustment")
        print("- Position deadzone configuration")
        print("- Arduino port selection")
        print("- Servo movement speed")
        print("- Laser duration settings")
        
        # Flash the settings button to show it was clicked
        original_text = self.settings_button.cget("text")
        self.settings_button.configure(text="⚙️ Coming Soon!")
        self.root.after(1500, lambda: self.settings_button.configure(text=original_text))
    
    def setup_gui(self):
        """Set up the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Video display label
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, pady=(0, 10))
        
        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=10)
        
        # First row of buttons
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(pady=5)
        
        # Start/Stop button
        self.toggle_button = ttk.Button(
            button_row1, 
            text="Stop Camera", 
            command=self.toggle_camera,
            width=15
        )
        self.toggle_button.pack(side=tk.LEFT, padx=5)
        
        # Toggle detection button
        self.detection_button = ttk.Button(
            button_row1,
            text="Disable Detection",
            command=self.toggle_detection,
            width=15
        )
        self.detection_button.pack(side=tk.LEFT, padx=5)
        
        # Toggle crosshair button
        self.crosshair_button = ttk.Button(
            button_row1,
            text="Hide Crosshair",
            command=self.toggle_crosshair,
            width=15
        )
        self.crosshair_button.pack(side=tk.LEFT, padx=5)
        
        # Manual shoot button
        self.shoot_button = ttk.Button(
            button_row1,
            text="🎯 SHOOT",
            command=self.manual_shoot,
            width=15
        )
        self.shoot_button.pack(side=tk.LEFT, padx=5)
        
        # Second row of buttons
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(pady=5)
        
        # Arduino reconnect button
        self.arduino_button = ttk.Button(
            button_row2,
            text="Reconnect Arduino",
            command=self.reconnect_arduino,
            width=15
        )
        self.arduino_button.pack(side=tk.LEFT, padx=5)
        
        # Test Arduino button
        self.test_arduino_button = ttk.Button(
            button_row2,
            text="Test Arduino",
            command=self.test_arduino_connection,
            width=15
        )
        self.test_arduino_button.pack(side=tk.LEFT, padx=5)
        
        # Settings button (placeholder for future settings)
        self.settings_button = ttk.Button(
            button_row2,
            text="⚙️ Settings",
            command=self.show_settings,
            width=15
        )
        self.settings_button.pack(side=tk.LEFT, padx=5)
        
        # Exit button
        exit_button = ttk.Button(
            button_row2, 
            text="❌ Exit", 
            command=self.on_closing,
            width=15
        )
        exit_button.pack(side=tk.LEFT, padx=5)
        
        # Status and detection info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, pady=10)
        
        # Status label
        self.status_label = ttk.Label(info_frame, text="Camera: Active")
        self.status_label.grid(row=0, column=0, padx=10)
        
        # Detection status label
        self.detection_status_label = ttk.Label(info_frame, text="Detection: ON")
        self.detection_status_label.grid(row=0, column=1, padx=10)
        
        # Detection count label
        self.count_label = ttk.Label(info_frame, text="Drones Detected: 0")
        self.count_label.grid(row=0, column=2, padx=10)
        
        # Arduino connection status
        arduino_status = "Connected" if self.arduino_connected else "Disconnected"
        self.arduino_status_label = ttk.Label(info_frame, text=f"Arduino: {arduino_status}")
        self.arduino_status_label.grid(row=1, column=0, padx=10)
        
        # Position feedback label
        self.position_label = ttk.Label(info_frame, text="Position: Center")
        self.position_label.grid(row=1, column=1, padx=10)
        
        # Serial port info
        port_info = f"Port: {self.serial_port}" if self.serial_port else "Port: None"
        self.port_label = ttk.Label(info_frame, text=port_info)
        self.port_label.grid(row=1, column=2, padx=10)
        
        # Confidence threshold frame
        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.grid(row=3, column=0, pady=10)
        
        ttk.Label(threshold_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=self.confidence_threshold)
        threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.1,
            to=1.0,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_threshold
        )
        threshold_scale.pack(side=tk.LEFT, padx=5)
        
        self.threshold_label = ttk.Label(threshold_frame, text=f"{self.confidence_threshold:.2f}")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
    
    def video_loop(self):
        """Main video capture and display loop"""
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if ret:
                    # Run YOLO detection if enabled and model is loaded
                    if self.detection_enabled and self.model is not None:
                        frame = self.run_detection(frame)
                    
                    # Draw crosshair and center square if enabled
                    if self.show_crosshair:
                        frame = self.draw_crosshair(frame)
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update the label with new image
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo  # Keep a reference
                else:
                    print("Error: Failed to capture frame")
                    break
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)  # ~30 FPS
    
    def run_detection(self, frame):
        """Run YOLO detection on the frame and draw bounding boxes"""
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Reset detection count and target acquisition for this frame
            frame_detections = 0
            self.target_acquired = False
            
            # Get frame center
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Draw bounding boxes and labels
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Calculate drone center
                        drone_center_x = (x1 + x2) // 2
                        drone_center_y = (y1 + y2) // 2
                        
                        # Calculate drone position relative to camera center
                        self.drone_position = self.calculate_drone_position(
                            drone_center_x, drone_center_y, width, height
                        )
                        
                        # Send motor commands to Arduino based on drone position
                        self.send_motor_commands(self.drone_position)
                        
                        # Check if drone is near center
                        distance_from_center = ((drone_center_x - center_x) ** 2 + (drone_center_y - center_y) ** 2) ** 0.5
                        is_centered = distance_from_center <= self.center_tolerance
                        
                        if is_centered:
                            self.target_acquired = True
                        
                        # Get class name (assuming your model has class names)
                        class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f"Class {class_id}"
                        
                        # Choose bounding box color based on whether it's centered
                        box_color = (0, 255, 0) if is_centered else (0, 255, 0)  # Green for detection
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Draw label with confidence
                        label = f"{class_name}: {confidence:.2f}"
                        if is_centered:
                            label += " [TARGETED]"
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), box_color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        # Draw line from drone center to crosshair center if targeted
                        if is_centered:
                            cv2.line(frame, (drone_center_x, drone_center_y), (center_x, center_y), (0, 0, 255), 2)
                            cv2.circle(frame, (drone_center_x, drone_center_y), 5, (0, 0, 255), -1)
                        
                        frame_detections += 1
            
            # Update detection count and position display
            if frame_detections > 0:
                self.detection_count = frame_detections
                self.count_label.configure(text=f"Drones Detected: {self.detection_count}")
                
                # Update position display
                position_text = f"Position: {self.drone_position['y'].title()}-{self.drone_position['x'].title()}"
                if self.drone_position['x'] == 'center' and self.drone_position['y'] == 'center':
                    position_text = "Position: Centered"
                self.position_label.configure(text=position_text)
            else:
                # No drone detected, reset position
                self.position_label.configure(text="Position: No Target")
            
        except Exception as e:
            print(f"Detection error: {e}")
        
        return frame
    
    def draw_crosshair(self, frame):
        """Draw crosshair lines and center square on the frame"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Choose crosshair color based on target acquisition
        current_color = self.crosshair_color_target if self.target_acquired else self.crosshair_color_normal
        
        # Draw vertical line
        cv2.line(frame, (center_x, 0), (center_x, height), current_color, self.crosshair_thickness)
        
        # Draw horizontal line
        cv2.line(frame, (0, center_y), (width, center_y), current_color, self.crosshair_thickness)
        
        # Draw center square
        half_size = self.center_square_size // 2
        top_left = (center_x - half_size, center_y - half_size)
        bottom_right = (center_x + half_size, center_y + half_size)
        cv2.rectangle(frame, top_left, bottom_right, current_color, self.crosshair_thickness)
        
        # Add center dot
        cv2.circle(frame, (center_x, center_y), 3, current_color, -1)
        
        # Draw targeting circle when target is acquired
        if self.target_acquired:
            cv2.circle(frame, (center_x, center_y), self.center_tolerance, (0, 255, 0), 2)
        
        # Add "SHOOT" text in top right corner when target is acquired
        if self.target_acquired:
            shoot_text = "SHOOT!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 3
            
            # Get text size
            text_size = cv2.getTextSize(shoot_text, font, font_scale, font_thickness)[0]
            
            # Position in top right corner
            text_x = width - text_size[0] - 20
            text_y = 40
            
            # Draw text background
            cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, shoot_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
            
            # Add blinking effect
            import time
            if int(time.time() * 3) % 2:  # Blink 3 times per second
                cv2.putText(frame, shoot_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    def toggle_crosshair(self):
        """Toggle crosshair display on/off"""
        self.show_crosshair = not self.show_crosshair
        if self.show_crosshair:
            self.crosshair_button.configure(text="Hide Crosshair")
        else:
            self.crosshair_button.configure(text="Show Crosshair")
    
    def manual_shoot(self):
        """Manual shoot action"""
        print("=== MANUAL SHOOT ACTIVATED ===")
        print("🎯 Target coordinates: Center of screen")
        print("🔥 Neutralization system engaged...")
        
        # Send shoot command to Arduino
        success = self.send_arduino_command("SHOOT")
        
        if success:
            print("✅ SHOOT command sent successfully to Arduino")
            # Flash the shoot button
            original_text = self.shoot_button.cget("text")
            self.shoot_button.configure(text="💥 FIRED!")
            
            # Reset button text after 1 second
            self.root.after(1000, lambda: self.shoot_button.configure(text=original_text))
        else:
            print("❌ Failed to send SHOOT command - Arduino not connected")
            # Flash button with error
            original_text = self.shoot_button.cget("text")
            self.shoot_button.configure(text="❌ ERROR!")
            
            # Reset button text after 1 second
            self.root.after(1000, lambda: self.shoot_button.configure(text=original_text))
        
        # Execute the shoot sequence
        self.execute_shoot_sequence()
    
    def execute_shoot_sequence(self):
        """Execute the actual shooting sequence"""
        # This is where you would implement the actual neutralization system
        # Examples:
        # - Activate servo motors to aim
        # - Trigger laser pointer
        # - Activate water cannon
        # - Send signal to drone jammer
        # - Log the shoot event
        
        print("Executing shoot sequence...")
        print("- Targeting system: LOCKED")
        print("- Neutralization method: ACTIVATED")
        print("- Shot fired at:", time.strftime("%H:%M:%S"))
        
        # You can add hardware control here
        # Example GPIO control (uncomment if using Raspberry Pi):
        # import RPi.GPIO as GPIO
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(18, GPIO.OUT)
        # GPIO.output(18, GPIO.HIGH)  # Activate shooting mechanism
        # time.sleep(0.5)
        # GPIO.output(18, GPIO.LOW)   # Deactivate
    
    def toggle_detection(self):
        """Toggle YOLO detection on/off"""
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.detection_button.configure(text="Disable Detection")
            self.detection_status_label.configure(text="Detection: ON")
        else:
            self.detection_button.configure(text="Enable Detection")
            self.detection_status_label.configure(text="Detection: OFF")
            self.detection_count = 0
            self.count_label.configure(text="Drones Detected: 0")
    
    def update_threshold(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = float(value)
        self.threshold_label.configure(text=f"{self.confidence_threshold:.2f}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.cap.isOpened():
            self.cap.release()
            self.toggle_button.configure(text="Start Camera")
            self.status_label.configure(text="Camera: Stopped")
            self.video_label.configure(image="")
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if self.cap.isOpened():
                self.toggle_button.configure(text="Stop Camera")
                self.status_label.configure(text="Camera: Active")
            else:
                print("Error: Could not restart camera")
                self.status_label.configure(text="Camera: Error")
    
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        
        # Close Arduino serial connection
        if self.arduino_serial:
            try:
                self.arduino_serial.close()
                print("Arduino connection closed")
            except:
                pass
        
        self.root.quit()
        self.root.destroy()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

import cv2
import imutils

class CameraHandler:
    def __init__(self, camera_index=0, width=1280, height=720, fps=30):
        self.cap = None
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            # Camera optimization
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self):
        """Read a frame from the camera"""
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if ret:
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
        return ret, frame
    
    def get_processed_frames(self):
        """Get both original and processed frames"""
        ret, frame = self.read_frame()
        if not ret:
            return False, None, None, None
            
        # Resize for processing efficiency
        frame_small = cv2.resize(frame, (640, 360))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        # Display frame
        display_frame = imutils.resize(frame, width=800)
        
        return True, frame, frame_rgb, display_frame
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            print("Camera released")
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def create_window(self, window_name="Smart Camera", width=1200, height=900):
        """Create and configure display window"""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
    
    def display_frame_with_status(self, frame, status_info, window_name="Smart Camera"):
        """Display frame with status overlay"""
        display_frame = frame.copy()
        
        # Status display
        status = status_info.get('status', 'Unknown')
        color = status_info.get('color', (255, 255, 255))
        
        cv2.putText(display_frame, status, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Additional info
        if status_info.get('gaze_detected'):
            cv2.putText(display_frame, "Gaze: CENTER", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if status_info.get('continuous_listening'):
            cv2.putText(display_frame, "🎤 LISTENING CONTINUOUSLY", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif status_info.get('listening_for_command'):
            cv2.putText(display_frame, "Listening for command...", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        cv2.putText(display_frame, "Press 'q' to quit | Say 'stop listening' to pause", 
                   (20, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, display_frame)
    
    def check_quit_key(self):
        """Check if quit key is pressed"""
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')
    
    def destroy_windows(self):
        """Destroy all OpenCV windows"""
        cv2.destroyAllWindows()
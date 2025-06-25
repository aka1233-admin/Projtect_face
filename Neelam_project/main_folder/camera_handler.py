"""
Threaded camera capture for smooth video processing
"""
import cv2
import threading
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS

class VideoCaptureThreaded:
    """Smooth Webcam Class using background thread"""
    def __init__(self, src=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)  
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        
        # Start the background thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
    
    def update(self):
        """Background thread to continuously capture frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
    
    def read(self):
        """Read the latest frame"""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)
    
    def release(self):
        """Release the camera and stop the thread"""
        self.running = False
        self.thread.join()
        self.cap.release()
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap.isOpened()
    
    def get_frame_dimensions(self):
        """Get current frame dimensions"""
        if self.frame is not None:
            return self.frame.shape
        return None
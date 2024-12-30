import cv2
import time

class Camera:
    def __init__(self, video_src):
        self.video_src = video_src
        if self.video_src is None:
            print('Video source not assigned, default webcam will be used')
            self.video_src = 0
            
        # Initialize capture with error handling
        self.cap = cv2.VideoCapture(self.video_src)
        if not self.cap.isOpened():
            raise ValueError(f'Failed to open video source: {self.video_src}')
            
        # Configure capture properties to minimize buffering
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Allow time for camera to initialize
        if isinstance(self.video_src, int):
            time.sleep(0.5)
            
        # Set optimal capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def get_frame_size(self):
        if not self.cap.isOpened():
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def get_frame(self):
        if not self.cap.isOpened():
            return False, None
            
        # Read frame with retry mechanism
        max_retries = 3
        for _ in range(max_retries):
            frame_got, frame = self.cap.read()
            if frame_got and frame is not None:
                # If the frame comes from webcam, flip it so it looks like a mirror.
                if isinstance(self.video_src, int):
                    frame = cv2.flip(frame, 2)
                return frame_got, frame
            time.sleep(0.1)  # Short delay before retry
            
        return False, None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            # Release resources
            self.cap.release()
            self.cap = None
            # Small delay to ensure proper cleanup
            time.sleep(0.1)
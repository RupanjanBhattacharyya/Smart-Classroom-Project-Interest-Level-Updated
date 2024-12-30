from argparse import ArgumentParser
import math
import time
import json
from pathlib import Path
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
import dlib

from src.pose_estimator import PoseEstimator
from src.landmark_metrics import eye_aspect_ratio, mouth_aspect_ratio
from src.eyes import update_eyes
from src.yawn import update_yawn
from src.video import Camera

print(__doc__)
print('OpenCV version: {}'.format(cv2.__version__))

# Parse arguments from user input
parser = ArgumentParser()
parser.add_argument('--video', type=str, default=None,
                    help='Video file to be processed.')
parser.add_argument('--cam', type=int, default=None,
                    help='The webcam index.')
args = parser.parse_args()


class LocalStorage:
    def __init__(self, session_name):
        self.data = []
        self.session_name = session_name
        self.output_dir = Path('Attention_data')
        self.output_dir.mkdir(exist_ok=True)
        
    def save(self, data_point):
        # Convert numpy types to native Python types before saving
        converted_data = {}
        for key, value in data_point.items():
            if isinstance(value, np.int64):
                converted_data[key] = int(value)
            elif isinstance(value, np.float64):
                converted_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                converted_data[key] = value.tolist()
            else:
                converted_data[key] = value
        self.data.append(converted_data)
    
    def generate_report(self):
        if not self.data:
            return {
                "session_summary": {
                    "start_time": "N/A",
                    "end_time": "N/A",
                    "duration_minutes": 0
                },
                "attention_metrics": {
                    "average_attention_percentage": 0,
                    "looking_away_duration_seconds": 0,
                    "looking_away_percentage": 0
                },
                "eye_metrics": {
                    "total_eyes_closed_seconds": 0,
                    "average_eyes_closed_duration": 0,
                    "maximum_eyes_closed_duration": 0
                },
                "yawn_metrics": {
                    "total_yawn_duration_seconds": 0,
                    "average_yawn_duration": 0,
                    "maximum_yawn_duration": 0
                },
                "head_pose_metrics": {
                    "average_angles": {
                        "x_angle": 0,
                        "y_angle": 0,
                        "z_angle": 0
                    }
                }
            }
        
        try:
            # Convert data to DataFrame for easier analysis
            df = pd.DataFrame(self.data)
            
            # Calculate session duration
            session_start = pd.to_datetime(df['timestamp'].iloc[0])
            session_end = pd.to_datetime(df['timestamp'].iloc[-1])
            session_duration = float((session_end - session_start).total_seconds() / 60)  # in minutes
            
            # Calculate attention metrics
            avg_attention = float(df['attention'].mean())
            
            # Calculate looking away metrics
            looking_away_duration = float(len(df[df['looking_away(head_pose)'] == True]) * 5)  # 5 seconds per data point
            looking_away_percentage = float((looking_away_duration / (len(df) * 5)) * 100)
            
            # Calculate eyes closed metrics
            total_eyes_closed = float(df['eyes_closed_duration'].sum())
            avg_eyes_closed = float(df['eyes_closed_duration'].mean())
            max_eyes_closed = float(df['eyes_closed_duration'].max())
            
            # Calculate yawn metrics
            total_yawn_duration = float(df['mouth_open_duration'].sum())
            avg_yawn_duration = float(df['mouth_open_duration'].mean())
            max_yawn_duration = float(df['mouth_open_duration'].max())
            
            # Calculate head pose metrics
            pose_data = pd.DataFrame([p if isinstance(p, list) else [0,0,0] for p in df['pose']], columns=['x', 'y', 'z'])
            avg_head_angles = {
                'x': float(pose_data['x'].mean()),
                'y': float(pose_data['y'].mean()),
                'z': float(pose_data['z'].mean())
            }
            
            report = {
                "session_summary": {
                    "start_time": session_start.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": session_end.strftime('%Y-%m-%d %H:%M:%S'),
                    "duration_minutes": round(session_duration, 2)
                },
                "attention_metrics": {
                    "average_attention_percentage": round(avg_attention, 2),
                    "looking_away_duration_seconds": round(looking_away_duration, 2),
                    "looking_away_percentage": round(looking_away_percentage, 2)
                },
                "eye_metrics": {
                    "total_eyes_closed_seconds": round(total_eyes_closed, 2),
                    "average_eyes_closed_duration": round(avg_eyes_closed, 2),
                    "maximum_eyes_closed_duration": round(max_eyes_closed, 2)
                },
                "yawn_metrics": {
                    "total_yawn_duration_seconds": round(total_yawn_duration, 2),
                    "average_yawn_duration": round(avg_yawn_duration, 2),
                    "maximum_yawn_duration": round(max_yawn_duration, 2)
                },
                "head_pose_metrics": {
                    "average_angles": {
                        "x_angle": round(avg_head_angles['x'], 2),
                        "y_angle": round(avg_head_angles['y'], 2),
                        "z_angle": round(avg_head_angles['z'], 2)
                    }
                }
            }
            
            return report
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return self.generate_report()  # Return empty report structure

    def print_report(self, report):
        """Print a formatted version of the attention report"""
        print("\nAttention Session Report")
        print("=" * 50)
        
        # Session Summary
        print("\nSession Summary:")
        print(f"Start Time: {report['session_summary']['start_time']}")
        print(f"End Time: {report['session_summary']['end_time']}")
        print(f"Duration: {report['session_summary']['duration_minutes']} minutes")
        
        # Attention Metrics
        print("\nAttention Metrics:")
        print(f"Average Attention: {report['attention_metrics']['average_attention_percentage']}%")
        print(f"Looking Away Duration: {report['attention_metrics']['looking_away_duration_seconds']} seconds")
        print(f"Looking Away Percentage: {report['attention_metrics']['looking_away_percentage']}%")
        
        # Eye Metrics
        print("\nEye Metrics:")
        print(f"Total Eyes Closed: {report['eye_metrics']['total_eyes_closed_seconds']} seconds")
        print(f"Average Eyes Closed Duration: {report['eye_metrics']['average_eyes_closed_duration']} seconds")
        print(f"Maximum Eyes Closed Duration: {report['eye_metrics']['maximum_eyes_closed_duration']} seconds")
        
        # Yawn Metrics
        print("\nYawn Metrics:")
        print(f"Total Yawn Duration: {report['yawn_metrics']['total_yawn_duration_seconds']} seconds")
        print(f"Average Yawn Duration: {report['yawn_metrics']['average_yawn_duration']} seconds")
        print(f"Maximum Yawn Duration: {report['yawn_metrics']['maximum_yawn_duration']} seconds")
        
        # Head Pose Metrics
        print("\nHead Pose Metrics (Average Angles):")
        print(f"X (Left/Right): {report['head_pose_metrics']['average_angles']['x_angle']}°")
        print(f"Y (Up/Down): {report['head_pose_metrics']['average_angles']['y_angle']}°")
        print(f"Z (Tilt): {report['head_pose_metrics']['average_angles']['z_angle']}°")
        
        print("\n" + "=" * 50)
    
    def save_to_file(self):
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = self.output_dir / f'{self.session_name}_{timestamp}.json'
            
            # Generate the report
            report = self.generate_report()
            
            # Create the final data structure
            output_data = {
                "session_data": self.data,  # Data is already converted in save() method
                "attention_report": report
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f'Session data and report saved to {filename}')
            
            # Print the report
            self.print_report(report)
        except Exception as e:
            print(f"Error saving to file: {str(e)}")
            
def draw_text(image, label, coords=(50,50)):
    cv2.putText(
        img=image,
        text=label,
        org=(coords[0], coords[1]),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 0, 255),
    )

def plot_attention_metrics(storage, attn_span):
    """
    Create visualizations of the attention tracking metrics and display them.
    """
    try:
        # Check if we have data to plot
        if attn_span.empty or not storage.data:
            print("No data available for plotting")
            return

        # Convert timestamp to datetime
        attn_span['timestamp'] = pd.to_datetime(attn_span['timestamp'])
        attn_span = attn_span.set_index('timestamp')

        # Create a figure with 3 subplots
        plt.close('all')  # Close any existing plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Attention Tracking Metrics Over Time', fontsize=16)

        # Convert stored data to DataFrame
        data_df = pd.DataFrame(storage.data)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df = data_df.set_index('timestamp')

        # Plot 1: Mouth Open Duration and Eyes Closed
        if 'mouth_open_duration' in data_df.columns:
            ax1.plot(data_df.index, data_df['mouth_open_duration'], 'm-', label='Yawn Duration (s)')
        ax1.set_ylabel('Yawn Duration (s)')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(attn_span.index, attn_span['eyes_closed'], 'r-', label='Eyes Closed (s)')
        ax1_twin.set_ylabel('Eyes Closed (s)', color='r')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Plot 2: Head Pose
        if 'pose' in data_df.columns:
            data_df['pose_x'] = data_df['pose'].apply(lambda x: x[0] if isinstance(x, list) else 0)
            data_df['pose_y'] = data_df['pose'].apply(lambda x: x[1] if isinstance(x, list) else 0)
            data_df['pose_z'] = data_df['pose'].apply(lambda x: x[2] if isinstance(x, list) else 0)
            
            ax2.plot(data_df.index, data_df['pose_x'], 'g-', label='X (Left/Right)')
            ax2.plot(data_df.index, data_df['pose_y'], 'b-', label='Y (Up/Down)')
            ax2.plot(data_df.index, data_df['pose_z'], 'r-', label='Z (Tilt)')
            ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(y=-20, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Head Pose Angles (degrees)')
            ax2.legend()

        # Plot 3: Attention Index
        ax3.plot(attn_span.index, attn_span['attention'], 'b-', label='Attention (%)')
        ax3.set_ylabel('Attention (%)')
        ax3.legend()

        # Format x-axis
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        plt.xlabel('Time')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('attention_metrics.png')
        print("Attention metrics plot saved as 'attention_metrics.png'")
        
        # Display the plot
        plt.show(block=False)
        
    except Exception as e:
        print(f"Error creating attention metrics plot: {str(e)}")

def run_attention_inference(video_src):
    global attn, looking_away, eyes_closed, time_eyes_closed, mouth_open, time_mouth_open
    
    # FPS
    prev_frame_time, cur_frame_time = 0, 0

    # Attention Metrics
    attn = 100
    attn_span = pd.DataFrame(columns=['timestamp', 'attention'])
    looking_away = None

    # Blink Detection
    eyes_closed, eyes_already_closed = False, False
    start_eyes_closed, time_eyes_closed = 0, 0

    # Yawn detection
    mouth_open, mouth_already_open = False, False
    start_mouth_open, time_mouth_open = 0, 0

    # Landmarks Detection
    pretrained_landmarks = r'assets/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pretrained_landmarks)

    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
    (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']
    # Initialize FPS variables

    prev_frame_time = 0
    cur_frame_time = 0
    
    cap = Camera(video_src)
    width, height = cap.get_frame_size()
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Initialize variables for local storage every 5 seconds
    poses = []
    avg_time_eyes_closed = 0
    avg_time_mouth_open = 0
    avg_attention = 0
    count = 0
    yawn_duration = 0
    last_saved = time.time()
    
    # Initialize local storage
    storage = LocalStorage('attention_session')
    
    # Initialize attention tracking DataFrame
    attn_span = pd.DataFrame(columns=['timestamp', 'attention', 'eyes_closed'])
    
    last_saved = time.time()
    
    try:
        while True:
            start = time.time()
            
            # Read a frame
            frame_got, frame = cap.get_frame()
            if frame_got is False:
                break

            # Converting the image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get faces into webcam's image
            rects = detector(gray, 0)

            cur_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')

            if not(rects):
                attn = max(attn-0.1,0)
                looking_away = None
                eyes_closed = False
                mouth_open = False
                pose = None
            else:
                # For each detected face, find the landmarks
                for (i, rect) in enumerate(rects):
                    # Make the prediction and transform it to numpy array
                    shape = predictor(gray, rect)
                    shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype='f')

                    # Estimate pose using the 68 facial landmarks
                    pose = pose_estimator.solve_pose_by_68_points(shape)
                    pose = pose[0]
                    pose = [ith_pose[0] * 180 / math.pi for ith_pose in pose]

                    print(f'[{cur_time}] (x,y,z): ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})')

                    # Attention Thresholds
                    face_left_right_threshold = 20
                    face_up_threshold = 30
                    face_down_threshold = 20
                    attn_change = 0.5
                    EAR_threshold = 0.25

                    # Eye Aspect Ratio (EAR)
                    left_eye = shape[left_eye_start:left_eye_end]
                    right_eye = shape[right_eye_start:right_eye_end]
                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)
                    EAR = (left_EAR + right_EAR) / 2.0
                    eyes_closed = EAR < 0.25
                    eyes_closed = EAR < EAR_threshold
                    print(f'LEFT: {left_EAR:.3f}, RIGHT: {right_EAR:.3f}, CLOSED: {eyes_closed}')

                    # Mouth Aspect Ratio (MAR)
                    mouth = shape[mouth_start:mouth_end]
                    MAR = mouth_aspect_ratio(mouth)
                    mouth_open = MAR > 0.3
                    print(f'YAWN: {MAR}')                    


                    time_eyes_closed, start_eyes_closed, eyes_already_closed = update_eyes(
                        eyes_closed, time_eyes_closed, eyes_already_closed, start_eyes_closed)

                    time_mouth_open, start_mouth_open, mouth_already_open = update_yawn(
                        mouth_open, time_mouth_open, mouth_already_open, start_mouth_open)

                    # Add/Deduct Attention based on the thresholds
                    if (-face_left_right_threshold < pose[0] < face_left_right_threshold) \
                        and (-face_down_threshold < pose[1] < face_up_threshold) \
                        and time_eyes_closed < 2 \
                        and time_mouth_open < 1:
                        attn = min(100, attn + attn_change / 2)
                        looking_away = False
                    else:
                        attn = max(attn-attn_change,0)
                        looking_away = True

                    print('-------------------------------------------------------------------------------')


            # Calculate FPS
            cur_frame_time = time.time()
            fps = 1/(cur_frame_time-prev_frame_time)
            prev_frame_time = cur_frame_time

            # Display metrics on the screen
            draw_text(frame, f'FPS: {int(fps)}', coords=(30,30))
            attentiveness = 'Please keep your face within the screen' if looking_away is None else f'Looking Away(Head Pose): {looking_away}'
            draw_text(frame, attentiveness, coords=(30,60))
            draw_text(frame, f'Attention: {attn:.2f}%', coords=(30,90))
            eyes_closed_text = f'{time_eyes_closed:.2f}s' if eyes_closed else ''
            draw_text(frame, f'Eyes Closed: {eyes_closed} {eyes_closed_text}', coords=(30,120))
            mouth_opened_text = f'{time_mouth_open:.2f}s' if mouth_open else ''
            draw_text(frame, f'Yawn: {mouth_open} {mouth_opened_text}', coords=(30,150))

            # Save data point every 5 seconds
            time_now = time.time()
            if time_now - last_saved > 5:
                last_saved = time_now
                data_point = {
                    'timestamp': cur_time,
                    'pose': pose if pose is not None else [0, 0, 0],
                    'attention': attn,
                    'eyes_closed_duration': time_eyes_closed,
                    'mouth_open_duration': time_mouth_open,
                    'looking_away(head_pose)': looking_away
                }
                storage.save(data_point)

            # Update attention span DataFrame
            attn_span = pd.concat([
                attn_span,
                pd.DataFrame({
                    'timestamp': [cur_time],
                    'attention': [attn],
                    'eyes_closed': [time_eyes_closed],
                })
            ], ignore_index=True)
            
            # Show preview
            cv2.imshow('Attention Tracker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC key
                break
            
    except KeyboardInterrupt:
            print("\nStopping attention tracking...")        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for i in range(4):
            cv2.waitKey(1)
        # Save data and create visualizations
        storage.save_to_file()
        plot_attention_metrics(storage, attn_span)
        plt.pause(0.1)  # Small pause to ensure plot window appears
        input("Press Enter to close...")  # Wait for user input before closing
        
    return storage, attn_span
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', type=str, default=None,
                        help='Video file to be processed.')
    parser.add_argument('--cam', type=int, default=None,
                        help='The webcam index.')
    args = parser.parse_args()
    
    try:
        video_src = args.cam if args.cam is not None else args.video
        storage, attn_span = run_attention_inference(video_src)
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    finally:
        cv2.destroyAllWindows()
        # Add extra calls to destroy windows
        for i in range(4):
            cv2.waitKey(1)
#!/usr/bin/python

import keras.backend as K
import sys
import math
import time
import json
from pathlib import Path
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from imutils import face_utils
import dlib

# Import attention tracking components
from src.pose_estimator import PoseEstimator
from src.landmark_metrics import eye_aspect_ratio, mouth_aspect_ratio
from src.eyes import update_eyes
from src.yawn import update_yawn
from src.video import Camera

# Import emotion detection components
from deployment.tensorflow_detector import *
from deployment.utils import label_map_util
from deployment.utils import visualization_utils_color as vis_util
from deployment.video_threading_optimization import *

print(__doc__)
print('OpenCV version: {}'.format(cv2.__version__))

# Emotion detection paths
PATH_TO_CKPT = 'deployment/frozen_graphs/frozen_inference_graph_face.pb'
PATH_TO_CLASS = 'deployment/frozen_graphs/classificator_full_model.pb'
PATH_TO_REGRESS = 'deployment/frozen_graphs/regressor_full_model.pb'
label_map = label_map_util.load_labelmap('deployment/protos/face_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


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
class InterestAnalyzer:
    def __init__(self):
        self.interest_history = []
        self.timestamps = []
    
    def calculate_interest_index(self, emotion_score, attention_percentage):
        """Calculate interest index using emotion score and attention percentage"""
        return 0.5 * emotion_score + 0.5 * (attention_percentage / 100)
    
    def get_interest_remark(self, interest_index):
        """Generate remark based on interest index"""
        if interest_index >= 0.8:
            return "Highly attentive with minimal signs of distraction."
        elif interest_index >= 0.6:
            return "Generally attentive with occasional distraction."
        elif interest_index >= 0.4:
            return "Moderately attentive with regular periods of distraction."
        elif interest_index >= 0.2:
            return "Frequently distracted with regular periods of distraction."
        else:
            return "Predominantly distracted with minimal attention."
    
    def generate_interest_report(self):
        if not self.interest_history:
            return "No interest data available"
        
        avg_interest = np.mean(self.interest_history)
        max_interest = max(self.interest_history)
        min_interest = min(self.interest_history)
        
        report = {
            "interest_metrics": {
                "average_interest_index": round(avg_interest, 3),
                "maximum_interest_index": round(max_interest, 3),
                "minimum_interest_index": round(min_interest, 3),
                "overall_engagement_level": self.get_interest_remark(avg_interest)
            }
        }
        return report

    def plot_interest_metrics(self, ax):
        """Plot interest metrics on the given axis"""
        if not self.interest_history:
            return
        
        timestamps = pd.to_datetime(self.timestamps)
        ax.plot(timestamps, self.interest_history, 'g-', label='Interest Index')
        ax.set_ylabel('Interest Index')
        ax.set_ylim([0, 1])
        ax.legend()


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

def run_integrated_inference(video_src):
    global attn, looking_away, eyes_closed, time_eyes_closed, mouth_open, time_mouth_open
    storage = None
    attn_span = None
    emotion_detector = None
    cap = None 

    interest_analyzer = InterestAnalyzer()
    
    try:
        # Initialize emotion detector
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        emotion_detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
        
        # Initialize attention tracking variables
        prev_frame_time, cur_frame_time = 0, 0
        fps = 0  # Default FPS value
        # Initialize attention tracking variables
        prev_frame_time = time.time()  # Initialize with current time
        cur_frame_time = prev_frame_time  # Initialize both to same value
        attn = 100
        looking_away = None
        eyes_closed, eyes_already_closed = False, False
        start_eyes_closed, time_eyes_closed = 0, 0
        mouth_open, mouth_already_open = False, False
        start_mouth_open, time_mouth_open = 0, 0

        # Landmarks Detection
        pretrained_landmarks = r'assets/shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(pretrained_landmarks)

        (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
        (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
        (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']

        # Initialize camera
        cap = Camera(video_src)
        if not cap.is_opened():
            raise ValueError(f"Failed to open video source: {video_src}")
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
        while True:
            frame_got, frame = cap.get_frame()
            if not frame_got or frame is None:
                print("Failed to grab frame, stopping...")
                break
            
            cur_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S.%f')

            try:

                # Process frame for attention tracking
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)



                # Process frame for emotion detection
                boxes, scores, classes, num_detections, emotion_score_detected, emotions_detected, emotion_index_detected = emotion_detector.run(frame)

                if not emotion_score_detected:
                    emotion_score_detected = [0.0]
                    emotions_detected = ["No emotion detected"]
                    emotion_index_detected = [0.0]

                if not(rects):
                    attn = max(attn-0.1,0)
                    looking_away = None
                    eyes_closed = False
                    mouth_open = False
                    pose = None
                else:
                    rect = rects[0]  # Process only the first face
                    # Process attention metrics
                    for (i, rect) in enumerate(rects):
                        shape = predictor(gray, rect)
                        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype='f')
                        
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

                # Calculate FPS with safeguard against division by zero
                cur_frame_time = time.time()
                time_diff = cur_frame_time - prev_frame_time
                fps = 1.0 / time_diff if time_diff > 0 else 0
                prev_frame_time = cur_frame_time

                # Draw metrics on frame
                draw_text(frame, f'FPS: {int(fps)}', coords=(30,20))
                attentiveness = 'Please keep your face within the screen' if looking_away is None else f'Looking Away(Head Pose): {looking_away}'
                draw_text(frame, attentiveness, coords=(30,40))
                draw_text(frame, f'Attention: {attn:.2f}%', coords=(30,60))
                eyes_closed_text = f'{time_eyes_closed:.2f}s' if eyes_closed else ''
                draw_text(frame, f'Eyes Closed: {eyes_closed} {eyes_closed_text}', coords=(30,80))
                mouth_opened_text = f'{time_mouth_open:.2f}s' if mouth_open else ''
                draw_text(frame, f'Yawn: {mouth_open} {mouth_opened_text}', coords=(30,100))
                draw_text(frame, f'Emotion: {emotions_detected}', coords=(30,120))
                draw_text(frame, f'Emotion Score: {emotion_score_detected}', coords=(30,140))
                draw_text(frame, f'Emotion Index: {emotion_index_detected}', coords=(30,160))
                
                # Calculate and display interest metrics
                interest_index = interest_analyzer.calculate_interest_index(
                    float(emotion_score_detected[0]), float(attn))
                interest_analyzer.interest_history.append(interest_index)
                interest_analyzer.timestamps.append(cur_time)
                    
                # Add interest metrics to the frame
                draw_text(frame, f'Interest Index: {interest_index:.2f}', coords=(30,180))
                draw_text(frame, f'Engagement Level: {interest_analyzer.get_interest_remark(interest_index)}', coords=(30,200))
                # Visualize emotion detection boxes
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1)
                
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
                cv2.imshow('Integrated Tracker', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC key
                    break

            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    except KeyboardInterrupt:
            print("\nStopping integrated tracking...")    
    except Exception as e:
        print(f"Error during inference: {e}")    
    finally:
        try:
            if cap is not None:
                cap.release()
            
            cv2.destroyAllWindows()
            for _ in range(4):  # Ensure windows are closed
                cv2.waitKey(1)
                
            # Generate reports if we have data
            if storage is not None:
                storage.save_to_file()
                
            if emotion_detector is not None:
                report = emotion_detector.generate_report()
                print("\nEmotion Detection Report:")
                print(report)
                plt.show()                
                fig = emotion_detector.plot_detector_results()
                if fig:
                    plt.savefig('emotion_analysis.png', dpi=300, bbox_inches='tight')
                    
            if storage is not None and attn_span is not None:
                plot_attention_metrics(storage, attn_span)
                plt.pause(0.1)  # Small pause to ensure plot window appears

            if interest_analyzer.interest_history:
                interest_report = interest_analyzer.generate_interest_report()
                print("\nInterest Level Report:")
                print(json.dumps(interest_report, indent=2))    
            
            if storage is not None:
                storage_data = storage.generate_report()
                combined_report = {
                    "attention_metrics": storage_data,
                    "interest_metrics": interest_report
                }
                
                # Save combined report
                report_path = storage.output_dir / f'{storage.session_name}_combined_report.json'
                with open(report_path, 'w') as f:
                    json.dump(combined_report, f, indent=2)
                print(f"\nCombined report saved to {report_path}")

        except Exception as e:
            print(f"Error during cleanup: {e}")

        return storage, attn_span

def draw_text(image, label, coords=(50,50)):
    cv2.putText(
        img=image,
        text=label,
        org=(coords[0], coords[1]),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        thickness=2,
        color=(0, 0, 255),
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--video', type=str, default=None,
                        help='Video file to be processed.')
    parser.add_argument('--cam', type=int, default=None,
                        help='The webcam index.')
    args = parser.parse_args()
    
    try:
        # Determine video source
        video_src = args.cam if args.cam is not None else args.video
        if video_src is None:
            print("No video source specified. Using default webcam (0)")
            video_src = 0
            
        # Run integrated inference
        storage, attn_span = run_integrated_inference(video_src)
        
    finally:
        # Final cleanup
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)
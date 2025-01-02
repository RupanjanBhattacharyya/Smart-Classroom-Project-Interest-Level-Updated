import time
import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.mobilenet import preprocess_input
from collections import Counter
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

emotion_classes = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Afraid', 'Disgusted', 'Angry', 'Contemptuous']

class TensorflowDetector(object):
    def __init__(self, PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS):
        self.emotion_history = []
        self.valence_history = []
        self.arousal_history = []
        self.emotion_scores = []  
        self.emotion_index = []

        # Initialize OpenCV's face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize session setup code remains exactly the same
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_1 = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

        self.classification_graph = tf.Graph()
        with self.classification_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CLASS, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.classification_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_2 = tf.compat.v1.Session(graph=self.classification_graph, config=config)

        self.regression_graph = tf.Graph()
        with self.regression_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_REGRESS, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.regression_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_3 = tf.compat.v1.Session(graph=self.regression_graph, config=config)
    
    def save_history_to_json(self):
        DATA_DIR = "emotion_data"
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # Fixed filename for consolidated report
        filepath = os.path.join(DATA_DIR, 'emotion_detection_session.json')
        current_report = self.generate_report()
        
        # Initialize or load existing data
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                history_data = json.load(f)
        else:
            history_data = {
                'sessions': [],
                'combined_metrics': {}
            }
        
        # Create current session data
        current_session = {
            'session_start': datetime.now().isoformat(),
            'session_data': [],
            'session_report': current_report
        }
        
        # Add entries to current session
        for i in range(len(self.emotion_history)):
            entry = {
                'timestamp': datetime.now().isoformat(),
                'emotion': self.emotion_history[i],
                'emotion_score': float(self.emotion_scores[i]),
                'emotion_index': float(self.emotion_index[i]),
                'valence': float(self.valence_history[i]),
                'arousal': float(self.arousal_history[i])
            }
            current_session['session_data'].append(entry)
        
        # Add current session to history
        history_data['sessions'].append(current_session)
        
        # Update combined metrics
        all_emotions = []
        all_scores = []
        all_indices = []
        all_valence = []
        all_arousal = []
        
        for session in history_data['sessions']:
            for entry in session['session_data']:
                all_emotions.append(entry['emotion'])
                all_scores.append(entry['emotion_score'])
                all_indices.append(entry['emotion_index'])
                all_valence.append(entry['valence'])
                all_arousal.append(entry['arousal'])
        
        history_data['combined_metrics'] = {
            'total_sessions': len(history_data['sessions']),
            'total_observations': len(all_emotions),
            'emotion_distribution': dict(Counter(all_emotions)),
            'average_scores': {
                'emotion_score': float(np.mean(all_scores)),
                'emotion_index': float(np.mean(all_indices)),
                'valence': float(np.mean(all_valence)),
                'arousal': float(np.mean(all_arousal))
            },
            'dominant_emotion': Counter(all_emotions).most_common(1)[0][0],
            'dominant_emotion_percentage': float(Counter(all_emotions).most_common(1)[0][1] / len(all_emotions))
        }
        # Save updated data
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=4)
        
        return filepath
    
    def calculate_emotion_score(self, prediction_row):
        return round(np.max(prediction_row), 4)

    def calculate_emotion_index(self, emotion, emotion_score, valence, arousal):
        """
        Calculate Emotion Index and apply emotion-based factor
        Positive factor for Happy, Neutral, Surprised
        Negative factor for all other emotions
        """
        valence_weight = 0.4
        arousal_weight = 0.6
        valence_norm = (valence + 1) / 2
        arousal_norm = (arousal + 1) / 2
        
        base_score = emotion_score * (valence_weight * valence_norm + arousal_weight * arousal_norm)
        
        # Apply emotion-based factor
        emotion_factor = 1 if emotion in ['Happy', 'Neutral', 'Surprised'] else -1
        final_score = base_score * emotion_factor
        
        return round(final_score, 4)

    def generate_report(self):
        if not self.emotion_history:
            return "No data collected for report generation"

        emotion_counts = Counter(self.emotion_history)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        dominant_emotion_score = emotion_counts[dominant_emotion] / len(self.emotion_history)
        
        avg_emotion_score = np.mean(self.emotion_scores)
        avg_emotion_index = np.mean(self.emotion_index)
        avg_valence = np.mean(self.valence_history)
        avg_arousal = np.mean(self.arousal_history)

        report = f"""
===Emotion Detection Report ===
Dominant Emotion: {dominant_emotion} ({dominant_emotion_score:.2%})
Average Emotion Score: {avg_emotion_score:.3f}
Average Emotion Index: {avg_emotion_index:.3f}

Detailed Metrics:
- Total Observations: {len(self.emotion_history)}
- Emotion Distribution: {dict(emotion_counts)}
- Average Valence: {avg_valence:.3f}
- Average Arousal: {avg_arousal:.3f}
========================
"""
        return report

    def plot_detector_results(self):
        if not self.emotion_history:
            print("No data available for plotting")
            return None

        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Emotion Distribution
        plt.subplot(2, 2, 1)
        emotion_counts = Counter(self.emotion_history)
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        plt.bar(emotions, counts)
        plt.title('Emotion Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')

        # Plot 2: Emotion Index Distribution
        plt.subplot(2, 2, 2)
        plt.hist(self.emotion_index, bins=20, color='skyblue', edgecolor='black')
        plt.title('Emotion Index Distribution')
        plt.xlabel('Emotion Index')
        plt.ylabel('Frequency')

        # Plot 3: Emotion vs Emotion Indexs
        plt.subplot(2, 2, 3)
        plt.scatter(self.emotion_scores, self.emotion_index, alpha=0.5)
        plt.xlabel('Emotion Confidence Score')
        plt.ylabel('Emotion Index')
        plt.title('Emotion vs Emotion Index Correlation')
        plt.grid(True)

        # Plot 4: Valence-Arousal Distribution
        plt.subplot(2, 2, 4)
        sc = plt.scatter(self.valence_history, self.arousal_history, 
                        c=self.emotion_index, cmap='RdYlGn', alpha=0.5)
        plt.colorbar(sc, label='Emotion Index')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Valence-Arousal Distribution')
        plt.xlabel('Valence')
        plt.ylabel('Arousal')

        plt.tight_layout()
        return fig

    def detect_faces_opencv(self, image):
   
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert OpenCV detections to the same format as TensorFlow
        h, w = image.shape[:2]
        
        if len(faces) == 0:
            # Return empty arrays with correct shapes if no faces detected
            return (np.array([[0, 0, 0, 0]]), 
                    np.array([0]), 
                    np.array([1]), 
                    np.array([0]))
        
        boxes = []
        scores = []
        
        for (x, y, w_face, h_face) in faces:
            # Convert to normalized coordinates
            ymin = y / h
            xmin = x / w
            ymax = (y + h_face) / h
            xmax = (x + w_face) / w
            
            boxes.append([ymin, xmin, ymax, xmax])
            scores.append(1.0)  # OpenCV doesn't provide confidence scores, so we use 1.0
        
        # Convert to numpy arrays with correct shapes
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.ones(len(scores))  # All detections are class 1 (face)
        num_detections = np.array(len(scores))
        
        # Reshape arrays to match TensorFlow format
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        classes = classes.reshape(-1)
        
        return boxes, scores, classes, num_detections

    def run(self, image):
        [h, w] = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image, axis=0)
        
        # Face detection with TensorFlow
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess_1.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        print('Detection time: {}'.format(round(time.time() - start_time, 8)))
        print('--------------------------------------')

        # Check if TensorFlow detection failed (no faces detected with confidence > 0.7)
        tf_detected = False
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        
        for i in range(min(20, boxes.shape[0] if len(boxes.shape) > 1 else 0)):
            if scores is None or scores[i] > 0.7:
                tf_detected = True
                break
        
        # If TensorFlow detection failed, use OpenCV
        if not tf_detected:
            print("TensorFlow detection failed, switching to OpenCV")
            boxes, scores, classes, num_detections = self.detect_faces_opencv(image)

        classification_input = self.classification_graph.get_tensor_by_name('input_1:0')
        classification_output = self.classification_graph.get_tensor_by_name('dense_2/Softmax:0')
        regression_input = self.regression_graph.get_tensor_by_name('input_1:0')
        regression_output = self.regression_graph.get_tensor_by_name('dense_2/BiasAdd:0')

        images_for_prediction = []
        for i in range(min(20, len(boxes))):
            if scores is None or scores[i] > 0.7:
                ymin, xmin, ymax, xmax = boxes[i]
                image_pred = image[max(int(h * ymin) - 20, 0):min(int(h * ymax) + 20, image.shape[:2][0]),
                            max(int(w * xmin) - 20, 0):min(int(w * xmax) + 20, image.shape[:2][1]), :]
                image_pred = Image.fromarray(image_pred).resize((224, 224))
                image_pred = keras.preprocessing.image.img_to_array(image_pred)
                image_pred = preprocess_input(image_pred)
                images_for_prediction.append(image_pred)

        emotions_detected = []
        emotion_index_detected = []
        emotion_score_detected = []
        
        if len(images_for_prediction) > 0:
            start_time = time.time()
            emotion_prediction = self.sess_2.run(classification_output,
                                            feed_dict={classification_input: images_for_prediction})
            print('Classification time: {}'.format(round(time.time() - start_time, 8)))
            
            start_time = time.time()
            va_prediction = self.sess_3.run(regression_output,
                                        feed_dict={regression_input: images_for_prediction})
            print('Regression time: {}'.format(round(time.time() - start_time, 8)))
            
            for idx, row in enumerate(emotion_prediction):
                pred = np.argmax(row)
                emotion = emotion_classes[pred]
                
                emotion_score = self.calculate_emotion_score(row)
                valence = va_prediction[idx][0]
                arousal = va_prediction[idx][1]
                emotion_index = self.calculate_emotion_index(emotion, emotion_score, valence, arousal)
                
                self.emotion_history.append(emotion)
                self.emotion_scores.append(emotion_score)
                self.emotion_index.append(emotion_index)
                self.valence_history.append(valence)
                self.arousal_history.append(arousal)
                
                emotions_detected.append(emotion)
                emotion_index_detected.append(emotion_index)
                emotion_score_detected.append(emotion_score)
                
                print(f"Emotion: {emotion} - Confidence Score: {emotion_score:.2f}")
                print(f"Emotion Index: {emotion_index:.2f}")
                print(f"Valence: {valence:.3f}, Arousal: {arousal:.3f}\n")

        # Save history to JSON after processing
        json_filename = self.save_history_to_json()
        print(f"Detection history saved to: {json_filename}")

        return boxes, scores, classes, num_detections, emotion_score_detected, emotions_detected, emotion_index_detected
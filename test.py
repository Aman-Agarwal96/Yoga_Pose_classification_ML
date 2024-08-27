import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score
import enum
import keras
import matplotlib.pyplot as plt

class BodyPart(enum.Enum):
  NOSE = 0
  LEFT_EYE = 1
  RIGHT_EYE = 2
  LEFT_EAR = 3
  RIGHT_EAR = 4
  LEFT_SHOULDER = 5
  RIGHT_SHOULDER = 6
  LEFT_ELBOW = 7
  RIGHT_ELBOW = 8
  LEFT_WRIST = 9
  RIGHT_WRIST = 10
  LEFT_HIP = 11
  RIGHT_HIP = 12
  LEFT_KNEE = 13
  RIGHT_KNEE = 14
  LEFT_ANKLE = 15
  RIGHT_ANKLE = 16


def get_center_point(landmarks, left_bodypart, right_bodypart):
    
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                BodyPart.RIGHT_HIP)
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    pose_center_new = tf.broadcast_to(pose_center_new,[tf.size(landmarks) // (17*2), 17, 2])
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,name="dist_to_pose_center")
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size



def normalize_pose_landmarks(landmarks):
    
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                BodyPart.RIGHT_HIP)

    pose_center = tf.expand_dims(pose_center, axis=1)
    pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
    landmarks = landmarks - pose_center
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    embedding = keras.layers.Flatten()(landmarks)
    return embedding


def preprocess_single_data(X_train):
    
    embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train), (1, 51)))
    embedding=tf.reshape(embedding, (34))
    return embedding

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(['filename'],axis=1, inplace=True)
    classes = df.pop('class_name').unique()
    y = df.pop('class_no')
    
    X = df.astype('float64')
    # y = keras.utils.to_categorical(y)
    return X, y, classes

def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train)


X_train, y_train, class_names = load_csv('train_data.csv')
processed_X_train = preprocess_data(X_train)
gbm_classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=3, verbose=True)

# Lists to store accuracy scores
train_accuracy_scores = []
val_accuracy_scores = []

# Training the model and collecting accuracy scores
for i in range(1):  # Iterate over each boosting iteration
    gbm_classifier.fit(X_train, y_train)  # Fit the model
    train_pred = gbm_classifier.predict(X_train)  # Predict on training data
    # val_pred = gbm_classifier.predict(X_val)  # Predict on validation data
    train_accuracy = accuracy_score(y_train, train_pred)  # Calculate training accuracy
    # val_accuracy = accuracy_score(y_val, val_pred)  # Calculate validation accuracy
    train_accuracy_scores.append(train_accuracy)  # Store training accuracy
    # val_accuracy_scores.append(val_accuracy)  # Store validation accuracy


plt.figure(figsize=(10, 6))
plt.plot(200, train_accuracy_scores)
plt.xlabel('Iterations')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy over Iterations')
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
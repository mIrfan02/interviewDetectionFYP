import time
from typing import Counter
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_wtf import FlaskForm
import librosa
import numpy as np
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from model_loader import load_model, video_model, fluency_model, feedback_model, summarization_model
from moviepy.editor import *
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import speech_recognition as sr
import os
from sqlalchemy import JSON,LargeBinary
from datetime import date

from flask_bcrypt import Bcrypt
from flask_bcrypt import check_password_hash
import pickle




app = Flask(__name__)
model = load_model()
video_model = video_model()
app.config['SECRET_KEY'] = '123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/fyp'
db = SQLAlchemy(app)
# model = keras.models.load_model('ver2.h5')

# Configure Flask-Login
loginmanager = LoginManager(app)
loginmanager.init_app(app)
# audio model


def preprocess_audio(data):
    # Add noise
    noise_amp = 0.04 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    print(data)
    # Stretch
    data = librosa.effects.time_stretch(data, rate=0.70)
    # Shift
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    data = np.roll(data, shift_range)
    return data


def check_emotions(emotion):
    if emotion == 0:
        return "Sad"
    elif emotion == 1:
        return "Disgust"
    elif emotion == 2:
        return "Angry"
    elif emotion == 3:
        return "Surprise"
    elif emotion == 4:
        return "Panic"
    elif emotion == 5:
        return "Happy"
    elif emotion == 6:
        return "Neutral"
    elif emotion == 7:
        return "Angry"
    elif emotion == 8:
        return "Disgust"
    elif emotion == 9:
        return "Panic"
    elif emotion == 10:
        return "Happy"
    elif emotion == 11:
        return "Neutral"
    elif emotion == 12:
        return "Sad"
    elif emotion == 13:
        return "Surprise"
    else:
        print(emotion)
        return "Invalid Emotion"
# Feature Extraction Function


def extract_features(data):
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# audion Model Prediction Function


def predict_emotion(model, features):
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=2)  # Add channel dimension
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    return emotion_index


def process_audio_chunks(audio_path, model):

    print("in Process audion chunks")
    # Load audio
    data, sr = librosa.load(audio_path, sr=22050)
    print("data", data)
    # Calculate number of samples in 10 seconds
    chunk_size = sr * 3

    # Initialize empty list to store emotion predictions
    emotion_predictions = []

    # Process audio in chunks
    for i in range(0, len(data), chunk_size):
        # Extract chunk
        chunk = data[i:i+chunk_size]

        # Preprocess chunk
        preprocessed_chunk = preprocess_audio(chunk)

        # Extract features
        features = extract_features(preprocessed_chunk)

        # Predict emotion
        emotion_index = predict_emotion(model, features)

        # Append predicted emotion to list
        emotion_predictions.append(emotion_index)

    return emotion_predictions

# Video Model


def face_emotion_recognition(model, path):
    # Load the Haar Cascade for face detection
    face_haar_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("model", model)
    cap = cv2.VideoCapture(path)

    # Set the frame processing rate (adjust based on your video FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0  # Counter to track processed frames
    analysis_interval = int(fps * 5)  # Analyze emotions every 5 seconds
    startTime = time.time()
    emotions = []
    while cap.isOpened():
        #   print("frame Count",frame_count," analysis_interval ",analysis_interval)
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame for emotion recognition only at analysis intervals
        if frame_count % analysis_interval == 0:
            #   print("chk1")
            height, width, _ = frame.shape
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #   print("chk2")
            # Detect faces
            faces = face_haar_cascade.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        #   print("faces",faces)
            try:
                for (x, y, w, h) in faces:
                    # Draw a rectangle around the detected face
                  #   print("check 2")
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)

                    # Extract and preprocess the face region
                    roi_gray = gray_image[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    image_pixels = img_to_array(roi_gray)
                    image_pixels = np.expand_dims(image_pixels, axis=0)
                    image_pixels /= 255.0
                  #   print("check 3")

                    # Make predictions using the loaded model
                  #   print("Image pexiles",image_pixels)
                    predictions = model.predict(image_pixels)
                  #   print("predections",predictions)
                    max_index = np.argmax(predictions[0])
                    emotion_detection = (
                        'angry', 'disgust', 'panic', 'happy', 'neutral', 'sad', 'surprise')
                    emotion_prediction = emotion_detection[max_index]
                  #   print("emotion prediction ",emotion_prediction)
                    emotions.append(emotion_prediction)
                    # # Display emotion prediction and confidence
                    # cv2.putText(frame, f"Emotion: {emotion_prediction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # cv2.putText(frame, f"Confidence: {np.max(predictions[0]) * 100:.2f}%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2_imshow(frame)
            except Exception as e:
                pass

        # Display the frame with or without emotion recognition (optional)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # Increment frame counter

    cap.release()
    cv2.destroyAllWindows()
    return emotions


def speech_text(path):
    recognizer = sr.Recognizer()
    audioFile = sr.AudioFile(path)
    with audioFile as source:
        audioText = recognizer.record(source)
    try:
        textData = recognizer.recognize_google(audioText)
    except sr.UnknownValueError:
        textData = "Could not understand audio"
    except sr.RequestError as e:
        textData = "Service is down"
    return textData

#   print(totalTime)

# end here video model
# User model


# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(50), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password = db.Column(db.String(60), nullable=False)

#     def get_id(self):
#         return self.email

class Usersall(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    history = db.Column(db.LargeBinary, nullable=True)

    def get_id(self):
        return self.email
# Registration Form



class RegistrationForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    email = StringField('email', validators=[DataRequired(), Email()])
    password = PasswordField('password', validators=[DataRequired()])
    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email = StringField('email', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired()])
    submit = SubmitField("Sign In")


def most_same(emotion_list):
    print("fucntional casl")
    # Count the occurrences of each emotion
    emotion_counts = Counter(emotion_list)

    # Find the emotion with the highest count
    most_common_emotion = max(emotion_counts, key=emotion_counts.get)
    print(most_common_emotion)

    return most_common_emotion


def handle_history(user, history_data):
    print("--------------------------------- history ----------------------")
    # print("history data : ", history_data)
    history = user.history
    # print("history : ",history)
    # print("history1 : ", type(history))
    history= pickle.loads(history)
    print("HIsotry :",history)
    # print("history2: ", history2)
    print("--------------------------------- history ----------------------")
    # if history is None:
    #     history = {}
    #     print("if None")
    if len(history.keys()) > 0:
        # if str(date.today()) in history.keys():
        if str(date.today()) in history.keys():
            print("before history append")
            print(history[str(date.today())])
            history[str(date.today())].append(history_data)
            print("history if ")
            print("after history append")
            print(history[str(date.today())])

        else:
            history[str(date.today())] = [history_data]
            print("history else")
    else:
        print("else2")
        # print("history[str(date.today())]  : ", history[str(date.today())])
        # print("historydata2:", history_data)
        history[str(date.today())] = [history_data]
        print("history[str(date.today())]  : ", history[str(date.today())])
        print("historydata2:", history_data)

    user.history = pickle.dumps(history)
    print("Hiroty of user ------------------------------")
    print("user1 :", user.history)
    # db.session.add(user)
    db.session.commit()











# Routes

@loginmanager.user_loader
def load_user(email):
    user = Usersall.query.filter_by(email=email).first()
    if user:
        return user
    return None

bcrypt = Bcrypt()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    try:
        form = RegistrationForm()
        existing_user = Usersall.query.filter_by(email=form.email.data).first()
        print("Existing user ", existing_user)
        if existing_user:
            print("User already exists")
            message = "User with the same email already exists!"
            return render_template("login.html", form=form, message=message)
        else:
            # Hash the password before storing it
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')

            # Create a new user and add it to the database
            new_user = Usersall(
                username=form.name.data, email=form.email.data, password=hashed_password,history=pickle.dumps({}))
            db.session.add(new_user)
            db.session.commit()

            flash('Account created successfully. You can now log in.', 'success')
            return render_template("login.html", form=form)

    except Exception as e:
        print("Error:", e)
        return render_template("login.html")


@app.route('/user')
@login_required
def user():
    return render_template('user.html')


@app.route('/history')
@login_required
def history():
    # Retrieve the authenticated user's data
    user_data = Usersall.query.filter_by(email=current_user.email).first()

    # Extract and unpickle the history data if available
    user_history = pickle.loads(user_data.history) if user_data and user_data.history else {}

    # Prepare a list to hold the feedback for each date
    feedback_list = []

    # Check if history data is available
    if user_history:
        # Iterate over the history data and extract feedback for each date
        for date, data in user_history.items():
            for entry in data:
                feedback_list.append({
                    'date': date,
                    'feedback': entry.get('feedback', 'No feedback available')
                })

    # Sort feedback_list by date (newest first)
    feedback_list.sort(key=lambda x: x['date'], reverse=True)

    # Pagination logic
    page = request.args.get('page', 1, type=int)
    per_page = 5
    total = len(feedback_list)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_feedback = feedback_list[start:end]

    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page

    return render_template('history.html', feedback_list=paginated_feedback, page=page, total_pages=total_pages)


@app.route('/test')
def test():
    return current_user.username


@app.route('/login', methods=["POST"])
def login():
    try:
        form = LoginForm()
        print(f"form email : {form.email.data}, password : {form.password.data}")
        user = Usersall.query.filter_by(email=form.email.data).first()
        print("user ", user)
        if user and check_password_hash(user.password, form.password.data):
            print("Logged in")
            flash('Logged in successfully!', 'success')
            login_user(user)

            if current_user.is_authenticated:
                print("Current user after login:", current_user.username)

            return redirect(url_for('user'))

        else:
            print("Invalid username or password")
            message = "Invalid username or password. Please try again."
            flash(message, 'danger')
            return render_template("login.html", form=form, message=message)

    except Exception as e:
        flash('An error occurred. Please try again later.', 'danger')
        return render_template("login.html")
# Login Route (placeholder, you can implement as needed)


@app.route('/loginpage')
def loginpage():
    print("login")
    return render_template('login.html')


@app.route('/logout')
def logout():
    print("in logout")
    if current_user.is_authenticated:
        print("Logging out user:", current_user.username)
        logout_user()
        print("User has been logged out.")
    else:
        print("No user is currently logged in.")
    return redirect(url_for('index'))
# Home Route (placeholder, you can implement as needed)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard.html')
@login_required
# dashboard route
def dashboard():

    return render_template('dashboard.html')

# User Profile


@app.route('/user.html')
@login_required
def userProfile():
    return render_template('user.html')

# Audio route


@app.route('/audio.html')
@login_required
def Audio():
    return render_template('audio.html')

# Video Analyzer route


@app.route('/video.html')
@login_required
def video():
    return render_template('video.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    print("function call")
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return 'No file part'

        audio_file = request.files['file']
        print(audio_file)

        # If the user does not select a file, the browser submits an empty file without a filename
        if audio_file.filename == '':
            return 'No selected file'

        if audio_file:
            print("audion file", audio_file)
            # Save the uploaded audio file to the desired location
            audio_path = audio_file.filename
            audio_file.save(audio_path)

            # Process the uploaded audio file
            emotion_index = process_audio_chunks(audio_path, model)
            print("type of emtion_index", type(emotion_index))
            emotions = []
            for i in emotion_index:
                emotions.append(check_emotions(i))
            print(emotions)
            return jsonify({"emotion": emotions})
    except Exception as e:
        return str(e)


# For video
@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return 'no file part'
        video_file = request.files['file']
        print(video_file)

        print("Curent user email  :  >>>>>>>>>>>>")
        print(current_user.email)

        if video_file:
            video_path = video_file.filename  # Adjust the path as necessary
            video_file.save(video_path)
            video_emotion = face_emotion_recognition(video_model, video_path)
            # most_repeated = most_same(video_emotion)
            print(video_emotion)
            audio_exists = seperate_audio(video_path)
            emotions = []
            global fluency
            global feedback
            global summary
            fluency = ""
            feedback = ""
            summary = ""
            if audio_exists == True:
                emotion_index = process_audio_chunks(
                    "seperated_audio.wav", model)
                text = speech_text("seperated_audio.wav")

                for i in emotion_index:
                    emotions.append(check_emotions(i))

                fluency = fluency_model(text)
                feedback = feedback_model(video_emotion, emotions)
                summary = summarization_model(text)
            most_same(video_emotion)
            most_common_emotion = most_same(video_emotion)
            loged_user = Usersall.query.filter_by(
                email=current_user.email).first()
            history = {
                "feedback": feedback,
                "fluency": fluency,
                "voic_emotions": emotions,
                "face_emotions": video_emotion,
                "summary": summary
            }
            if loged_user:
                handle_history(loged_user, history)

            return {
                "audio": emotions,
                "video": video_emotion,
                "most_common_emotion": most_common_emotion,
                "fluency": fluency,
                "feedback": feedback,
                "summary": summary
            }
    except Exception as e:
        print("Error:", str(e))
        return str(e)

# video audio separation


def seperate_audio(video_path):
    print("video-path", video_path)
    video = VideoFileClip(video_path)
    print("video audio", video.audio)
    if video.audio:
        print(video)  # 2.
        audio = video.audio  # 3.
        print("audio ", audio)
        audio.write_audiofile("seperated_audio.wav")
        print("testing", audio)
        return True
    else:
        return False


if __name__ == '__main__':
    # with app.app_context():
    #     db.create_all()
    app.run(debug=True)

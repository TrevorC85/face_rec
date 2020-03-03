# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import picamera
import picamera.array
import numpy as np
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import io
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}




# Get a reference to the Raspberry Pi camera.
# If this fails, make sure you have a camera connected to the RPi and that you
# enabled your camera in raspi-config and rebooted first.
#camera = picamera.PiCamera()
#camera.resolution = (1280, 720)
#output = np.empty((1280, 720, 3), dtype=np.uint8)


#image = (image * 255).round().astype(np.uint8)
 

# Load a sample picture and learn how to recognize it.
print("Loading known face image(s)")

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
 
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            print ('dir not found')
            print (os.listdir(train_dir))
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            print (str(img_path))
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.

                print("Image not suitable for training")
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf
    
# def show_prediction_labels_on_image(img_path, predictions):
    # """
    # Shows the face recognition results visually.

    # :param img_path: path to image to be recognized
    # :param predictions: results of the predict function
    # :return:
    # """
    # pil_image = Image.open(img_path).convert("RGB")
    # draw = ImageDraw.Draw(pil_image)

    # for name, (top, right, bottom, left) in predictions:
        # # Draw a box around the face using the Pillow module
        # draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # # There's a bug in Pillow where it blows up with non-UTF-8 text
        # # when using the default bitmap font
        # name = name.encode("UTF-8")

        # # Draw a label with a name below the face
        # text_width, text_height = draw.textsize(name)
        # draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        # draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # # Remove the drawing library from memory as per the Pillow docs
    # del draw
 
    # # Display the resulting image
    # pil_image.show()

def predict(captured, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """


    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    
    # Load image file and find face locations
    #X_img = face_recognition.load_image_file(captured)
    X_face_locations = face_recognition.face_locations(captured)
    
    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        print ('no face found')
        return []
    print ('face found')
    #captured2 = Image.fromarray(captured).convert("RGB")
    #captured2.show()
    #time.sleep(10)
    
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(captured, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
     
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print (are_matches)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def netcap(classifier, video_capture):   
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    predictions = []
    while 0 == len(predictions):
        ret, frame = video_capture.read()
        print ('connected to cam?')
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                predictions = predict(img, classifier, model_path=None, distance_threshold=0.6)
            #frame = show_prediction_labels_on_image(frame, predictions)
            for name, (top, right, bottom, left) in predictions:
                print (predictions)
                print("- Found {} at ({}, {})".format(name, left, top))    
            
            if ord('q') == cv2.waitKey(10):
                cap1.release()
                cv2.destroyAllWindows()
                exit(0)
    

                
def CamraCap(classifier, video_capture):

    predictions = []
    #camera.start_preview(resolution=(320, 240))
    with picamera.PiCamera(framerate=5) as camera:
        camera.brightness = 55
        camera.contrast = 55
        time.sleep(2)
        camera.resolution = (1080, 720)
        camera.start_preview(resolution=(320, 240))
        while 0 == len(predictions):
            print("Capturing image.")
            with picamera.array.PiRGBArray(camera) as stream:
                camera.capture(stream, format='RGB')
                # At this point the image is available as stream.array
                image = stream.array

            # camera.capture(output, format="rgb")
            # captured = Image.fromarray(output).convert("RGB")
            predict(image, classifier, model_path=None, distance_threshold=0.6)
            # Find all the faces and face encodings in the current frame of video
            for name, (top, right, bottom, left) in predictions:
                print (predictions)
                print("- Found {} at ({}, {})".format(name, left, top))    
                
     
    camera.stop_preview()
    return name
    


classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=None)

video_capture = cv2.VideoCapture('rtsp://user:password@192.168.1.103:554')

#name = netcap(classifier, video_capture)
name = CamraCap(classifier, video_capture)



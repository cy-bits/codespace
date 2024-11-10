import cv2
import face_recognition
import numpy as np
import pickle
import os

def capture_samples(cap_sample=5):
    cap = cv2.VideoCapture(0)
    encodings = []
    sample_count = 0

    while sample_count < cap_sample:
        ret, frame = cap.read()
        cv2.imshow("Capturing Samples", frame)
       
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
           
            face_location = face_recognition.face_locations(frame)
            face_encoding = face_recognition.face_encodings(frame, face_location)

            if face_encoding:
                encodings.append(face_encoding[0])
                sample_count += 1
                print(f"Sample captured {sample_count} out of {cap_sample}.")
                #name1 = name + str(sample_count)
                #image_path = os.path.join("images/", f"{name1}.jpg")
                #cv2.imwrite(image_path, frame)
                #print(f"Image saved as {image_path}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return encodings

#2. save_encodings
#This function saves the person’s name and averaged encoding to a pickle file.

def save_encodings(name, encodings, data_file="face_data.pkl"):
    average_encoding = np.mean(encodings, axis=0)
    '''
    try:
        with open(data_file, "rb") as f:
            face_data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        face_data = {}
    '''
    face_data = {}
    face_data[name] = average_encoding

    with open(data_file, "ab") as f:
        pickle.dump(face_data, f)

    print(f"Encodings for {name} saved successfully.")


#3. load_encodings
#This function loads all stored face encodings from the pickle file.

def load_encodings(data_file="face_data.pkl"):
    face_data = {}
    f = open(data_file, "rb")
    try:
        print('Loading face encodings...')
        while True:
                face_data.update(pickle.load(f))
    except EOFError:
        pass
    except FileNotFoundError:
        face_data = {}

    return face_data

#4. recognize_faces
#This function takes in known encodings and names, along with a frame, and returns detected face names and their positions.

def recognize_faces(known_face_encodings, known_face_names, frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

#5. draw_label
#This function draws a rectangle and label around each detected face.

def draw_label(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


#6. run_face_recognition
#This is the main function that ties everything together to run real-time face recognition.

def run_face_recognition(data_file="face_data.pkl"):
    face_data = load_encodings(data_file)
    known_face_names = list(face_data.keys())
    known_face_encodings = list(face_data.values())

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, face_names = recognize_faces(known_face_encodings, known_face_names, frame)
        frame = draw_label(frame, face_locations, face_names)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def delete_record(key):
    face_data = load_encodings()
    if key in face_data:
        del face_data[key]
        print(f"Record with key '{key}' Matched and Deleted.")
        with open("face_data.pkl", "wb") as f:
            pickle.dump(face_data, f)
            print("Records Deleted Successfully")
        f.close()
    else:
        print(f"Record for '{key}' not found in the dataset.")


while True:
    print('''
    1. Capture Sample
    2. Run Real-time Face Detection
    3. Read Data
    4. Delete Data
    0. Exit
    ''')
    ch = int(input('\nEnter choice: '))
    if ch == 1:
        name = input('Enter person name: ')
        encodings = capture_samples()
        if encodings:
            save_encodings(name, encodings)
        continue
    elif ch == 2:
        run_face_recognition()
        continue
    elif ch == 3:
        print(load_encodings())
        continue
    elif ch == 4:
        key = input('\nEnter name to delete face data: ')
        delete_record(key)
        continue
    elif ch == 0:
        print('\nThank You.\nByee...')
        break
    else:
        print('\nInvalid choice')
        continue
    


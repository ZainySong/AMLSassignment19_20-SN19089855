# README -- AMLS Assignment
Applied Machine Learning Systems ELEC0134 (19/20) Assignment

SN: 19089855

## The organization of the project
The organization of the codes is under the requirement.
### The assignment tasks
A. Binary tasks (celeba dataset)
A1: Gender detection: male or female.
A2: Emotion detection: smiling or not smiling.

B. Multiclass tasks (cartoon_set dataset)
B1: Face shape recognition: 5 types of face shapes.
B2: Eye color recognition: 5 types of eye colors.
### How to use the code
Run the main.py

## The role of each file

* main.py 

The operation file to run all models.

*data_preprocessing_A.py

Pre-process the data for task A1 and task A2

*data_preprocessing_B.py

Pre-process the data for task B1 and task B2

*landmarks.py

Extract the face features from the images.

*gender_model.py, emotion_model.py, face_model.py and eye_model.py

Each model file in each different folder is the model for each different task.

## The packages required to run the code

Numpy == 1.16.4, 

Pandas == 0.24.2,

matplotlib == 3.1.0, 

dlib == 19.17.0, 

opencv == 4.1.2, 

scikit-learn == 0.21.2, 

scikit-image == 0.15.0, 

keras == 2.2.4, 

tensorflow == 1.13.2




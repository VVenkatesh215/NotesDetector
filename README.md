# NotesDetector
During exams, students often share notes via social media, but manually deleting these images after the exam is time-consuming. To solve this problem, I developed a system that automatically detects handwritten notes (captured using a mobile camera) and deletes them efficiently.

---

## Getting Started
The system contains:
- **4 Python files**:
  - `train_network.py`: Script to train the model.
  - `test_network.py`: Script to test the model on individual images.
  - `delete_images.py`: Script to classify and delete "notes" images from a folder.
  - `lenet.py`: Implementation of the LeNet model.
- **1 Trained Model File**:
  - `mymodel.keras`: Pre-trained model for immediate use.
- **2 Folders**:
  - `test_images/`: Folder for testing images.
  - `backup_images/`: Folder to store classified "notes" images which are about to delete.

---

## Prerequisites
To run this project, you need the following Python libraries:
- **Keras**: Deep learning framework.
- **TensorFlow**: Backend for Keras.
- **OpenCV (cv2)**: For image processing.
- **NumPy**: For numerical computations.
- **scikit-learn**: For data splitting and evaluation.
- **Matplotlib**: For visualizing training results.
- **imutils**: For image utilities.


# Installing
I have already trained the model so you can direcly use it (mymodel.keras), so you can directly use it without training. However, if you want to train your own model, follow the instructions below.

Training the Model
Add your training images to the following folders:

images/notes/: Images containing notes.

images/not_notes/: Images not containing notes.

Run the training script:

python train_network.py
This will train the model and save it as mymodel.keras.

Testing the Model
Add your testing images to the test_images/ folder.

Test the model on individual images using:

python test_network.py --image test_images/image_name.jpg
The script will display the classification result (e.g., "notes" or "not_notes") and the confidence score.

Deleting "Notes" Images
To automatically delete "notes" images from a folder, modify the path in delete_images.py to point to your desired folder.

Run the script:

python delete_images.py
The script will classify images in the folder and move "notes" images to the backup_images/ folder.

## Built With
Keras: Deep learning framework used for building and training the model.

LeNet: Convolutional Neural Network architecture used for image classification.

OpenCV: For image processing and manipulation.

Matplotlib: For visualizing training and testing results.



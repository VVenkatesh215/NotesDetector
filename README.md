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

### Prerequisites
To run this project, you need the following Python libraries:
- **Keras**: Deep learning framework.
- **TensorFlow**: Backend for Keras.
- **OpenCV (cv2)**: For image processing.
- **NumPy**: For numerical computations.
- **scikit-learn**: For data splitting and evaluation.
- **Matplotlib**: For visualizing training results.
- **imutils**: For image utilities.


### Installing
I have already trained the model so you can direcly use it (mymodel.keras), so you can directly use it without training.
```
mymodel.keras
```
However, if you want to train your own model, follow the instructions below.

Training the Model
You can also train you own network by running "train_network.py" to train you own network you have to add images to images/notes and images/not_notes (Actually I trained the model with my personal images so I did not upload it)

```
run: python train_network.py
```

Testing the Model
Add your testing images to the test_images/ folder.

Test the model on individual images using:
```
run: python test_network.py --image test_images/image_name.jpg
```

Deleting "Notes" Images
To automatically delete "notes" images from a folder, modify the path in delete_images.py to point to your desired folder.

```
run: python delete_images.py
```

## Built With
[Keras](https://keras.io/): Deep learning framework used for building and training the model.

[LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf/): Convolutional Neural Network architecture used for image classification.

[OpenCV](https://docs.opencv.org/4.x/index.html): For image processing and manipulation.

[Matplotlib](https://matplotlib.org/stable/index.html): For visualizing training and testing results.

## Examples
<img src="https://github.com/VVenkatesh215/NotesDetector/blob/master/testresult1.png">
<img src="https://github.com/VVenkatesh215/NotesDetector/blob/master/testresult2.png">



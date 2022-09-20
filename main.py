import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.applications import ResNet152 as EncoderModel
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization, Input
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os
import absl.logging
import re
import random
from bs4 import BeautifulSoup
from tqdm import tqdm

####################################################################################################
###> Remove warnings & info message...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)

####################################################################################################
###> Programm variables

# ACTIONS
TODO = [ 
    "from_scratch",
    "preprocess", 
    "train", 
    "evaluate", 
    "test", 
]

# DATASETS
DATASET_FOLDER     = "datasets/"
IMAGES_FOLDER      = DATASET_FOLDER + "images/"
ANNOTATIONS_FOLDER = DATASET_FOLDER + "annotations/"
ALLOWED_EXTENSIONS = [ ".jpg", ".jpeg", ".png" ]
TRAINING_RATIO     = .8
VALIDATION_RATIO   = .2
IMAGE_SIZE         = (224, 224)

# SAVES
SAVES_PATH                          = "./saves/"
GRAPHS_PATH                         = SAVES_PATH + "graphs/"
GRAPHS_TRAINING_LOSS_FILE_NAME      = "training_loss_history.png"
GRAPHS_TRAINING_ACCURACY_FILE_NAME  = "training_accuracy_history.png"
CHECKPOINTS_PATH                    = SAVES_PATH + "checkpoints/"
CHECKPOINTS_FILE_NAME               = "best_weights"

# DETECTION
MAX_DETECTABLE_OBJECTS = 20
MAX_POINTS_BY_OBJECT   = 4
MAX_POINTS_BY_IMAGE    = MAX_DETECTABLE_OBJECTS * MAX_POINTS_BY_OBJECT

# TRAIN
TRAINING_PATIENCE   = 10
EPOCHS              = 250
BATCH_SIZE          = 32
DROPOUT             = .5
LEARNING_RATE       = 0.001

# TEST
NUMBER_OF_IMAGES_TO_TEST    = 5
MIN_ID_TO_TEST              = 1
MAX_ID_TO_TEST              = 50
IMAGE_IDS_TO_TEST           = [ random.randint(MIN_ID_TO_TEST, MAX_ID_TO_TEST) for _ in range(NUMBER_OF_IMAGES_TO_TEST)]

####################################################################################################
###> Launching the programm

print()
print("Starting...")
print("Actions todo: ", TODO)
print()

####################################################################################################
###> Clean previously saved files

if "from_scratch" in TODO:

    print("Removing files...")
    def removeFilesMatching(path):
        files = glob.glob(path)
        for file in files:
            os.remove(file)
        print("%d files removed matching pattern %s" % (len(files), path))
    removeFilesMatching(CHECKPOINTS_PATH + "*")
    removeFilesMatching(GRAPHS_PATH + "*")

if "train" in TODO:
    def createFolderIfNotExists(folder):
        if not(os.path.isdir(folder)):
            os.makedirs(folder)
    createFolderIfNotExists(CHECKPOINTS_PATH)
    createFolderIfNotExists(GRAPHS_PATH)

####################################################################################################
###> Filter images

if "preprocess" in TODO:

    def isCorrupted(fileimage):

        # with open(fileimage, "rb") as fobj:
        #     if not tf.compat.as_bytes("JFIF") in fobj.peek(10):
        #         return True

        try:
            with Image.open(fileimage) as img:
                img.verify()
            return False
        except:
            return True

    def removeInvalidImages(folder):

        num_total   = 0
        num_skipped = 0

        # Foreach images
        for f in os.listdir(folder):
            file = os.path.join(folder, f)
            if os.path.isfile(file):
                if not(
                    any([file.endswith(ext) for ext in ALLOWED_EXTENSIONS]) 
                        and 
                    not(isCorrupted(file))
                ):
                    os.remove(file)
                    num_skipped += 1
                    
                num_total += 1

        print("\nRemove bad formatted files...")
        print("Deleted %d / %d invalid images" % (num_skipped, num_total))

    removeInvalidImages(IMAGES_FOLDER)

####################################################################################################
###> Generate the datasets

def getPathsByIdFromFolder(folder):

    paths = {}

    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            currentId = list(map(int, re.findall(r'\d+', f)))[-1]
            paths[currentId] = full_path
    
    return paths

DATASETS_BOX_PATHS = getPathsByIdFromFolder(IMAGES_FOLDER)
DATASETS_ANNOTATIONS_PATHS = getPathsByIdFromFolder(ANNOTATIONS_FOLDER)

def getPathsFromId(id):

    imagePath      = DATASETS_BOX_PATHS[id] if id in DATASETS_BOX_PATHS else None
    annotationPath = DATASETS_ANNOTATIONS_PATHS[id] if id in DATASETS_ANNOTATIONS_PATHS else None

    if not(imagePath) or not(annotationPath):
        print("\nWarning: Can't find the image with the id" + str(image_id_to_test))
        return None

    return imagePath, annotationPath

def extractDataFromId(id):

    image_path, annotation_path = getPathsFromId(id)
    with open(annotation_path, 'r') as annotation_file:
        annotation_data = BeautifulSoup(annotation_file.read(), "xml")
        
    # Extract image
    image = keras.utils.load_img(image_path)
    image_w = int(annotation_data.find('width').text)
    image_h = int(annotation_data.find('height').text)

    # Image processing
    processed_image = keras.utils.img_to_array(image.resize(IMAGE_SIZE))

    # Extract boxes
    boxes_data = annotation_data.find_all('bndbox')
    boxes = [ (int(box.find('xmin').text), int(box.find('ymin').text), int(box.find('xmax').text), int(box.find('ymax').text)) for box in boxes_data ]
    coords = [ coord for box in boxes for coord in box ]

    # Boxes processing

    #> Store position in %
    full_percentage_boxes = [ ( float(min_x) / image_w, float(min_y) / image_h, float(max_x) / image_w, float(max_y) / image_h ) for min_x, min_y, max_x, max_y in boxes ]
    full_percentage_coords = [ full_percentage_coord for full_percentage_box in full_percentage_boxes for full_percentage_coord in full_percentage_box ]

    #> Complete with empty boxes
    empty_percentage_coords = [ 0 for i in range(len(full_percentage_coords), MAX_POINTS_BY_IMAGE) ]
    processed_percentage_coords =  full_percentage_coords + empty_percentage_coords

    return ( ( image, coords ), ( processed_image, processed_percentage_coords ) )

def plotRectangle(min_x, min_y, max_x, max_y, color, linestyle="solid"):

    plt.plot(
        [ min_x, max_x, max_x, min_x, min_x ],
        [ min_y, min_y, max_y, max_y, min_y ], 
        color=color, linestyle=linestyle
    )

def displayImageWithTargets(img, tgts=None, preds=None, show=True, figure_name="Image with targets"):
    
    plt.figure(figure_name)
    plt.imshow(img)

    # Image dimentions
    image_w, image_h = image.size[:2]
    plotRectangle(0, 0, image_w, image_h, color="grey", linestyle="solid")

    # Targets
    if not(tgts is None):
        for i in range(len(tgts)//4):
            min_x, min_y, max_x, max_y = tgts[i*4:i*4+4]
            plotRectangle(min_x, min_y, max_x, max_y, color="white", linestyle="dashed")

    # Pred
    if not(preds is None):
        for i in range(len(preds)//4):
            min_x, min_y, max_x, max_y = preds[i*4:i*4+4]
            plotRectangle(min_x, min_y, max_x, max_y, color="red", linestyle="dashed")

    if show:
        plt.show()

if "train" in TODO or "evaluate" in TODO:

    print("\nLoading the datasets...")

    images  = []
    targets = []
    ids = list(DATASETS_BOX_PATHS.keys())
    random.shuffle(ids)

    for id in tqdm(ids):

        ( ( image, coords ), ( processed_image, processed_coords ) ) = extractDataFromId(id)
        images.append(processed_image)
        targets.append(processed_coords)

        # displayImageWithTargets(image, coords, figure_name="1", show=False)
        # plt.show()

        pass
    
    # Separate training and testing data thanks to r1 ratio
    # Convert the list to numpy array, split to train and test dataset
    r = int(len(images)*TRAINING_RATIO)

    if "train" in TODO:
        (x_train), (y_train) = ( np.asarray(images[:r]),   np.asarray(targets[:r])   )
        print("Working with %s images, including %s for training and %s for validation" % (len(images), int(len(x_train)*(1-VALIDATION_RATIO)), int(len(x_train)*VALIDATION_RATIO)))
    
    if "evaluate" in TODO:
        (x_test),  (y_test)  = ( np.asarray(images[r:]),   np.asarray(targets[r:])   )
        print("Working with %s images, including %s for tests" % (len(images), len(x_test)))

####################################################################################################
###> Build a model

if "train" in TODO or "evaluate" in TODO or "test" in TODO:
    
    def make_model(input_shape):
        
        encoder = EncoderModel(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
        encoder.trainable = False

        i = encoder.input
        o = encoder.output
        o = Flatten()(o)
        o = Dense(256, activation="relu")(o)
        o = Dense(128, activation="relu")(o)
        o = Dense(MAX_POINTS_BY_IMAGE, activation="sigmoid")(o)

        # construct the model we will fine-tune for bounding box regression
        return Model(inputs=i, outputs=o)

    model = make_model(input_shape=IMAGE_SIZE + (3,))
    # keras.utils.plot_model(model, show_shapes=True)

    pass

####################################################################################################
###> Train the model

if "train" in TODO or "evaluate" in TODO or "test" in TODO:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["accuracy"],
    )

    # keras.utils.plot_model(model, show_shapes=True)

    pass

if "train" in TODO:
    
    print("\nTraining the model...")

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_RATIO,
        callbacks=[
            keras.callbacks.ModelCheckpoint(CHECKPOINTS_PATH + CHECKPOINTS_FILE_NAME, save_best_only=True, save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=TRAINING_PATIENCE),
        ],
    )

    def visualizeLearningHistory(history, show=True):

        h = history.history
  
        plt.figure("Loss history")
        plt.plot(h['loss'],         color='red', label='Train loss')
        plt.plot(h['val_loss'],     color='green', label='Val loss')
        plt.legend()
        plt.title('Training and validation loss over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(GRAPHS_PATH + GRAPHS_TRAINING_LOSS_FILE_NAME)

        plt.figure("Accuracy history")
        plt.plot(h['accuracy'],     color='red',   label='Train accuracy')
        plt.plot(h['val_accuracy'], color='green', label='Val accuracy')
        plt.legend()
        plt.title('Training and validation accuracy over the time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(GRAPHS_PATH + GRAPHS_TRAINING_ACCURACY_FILE_NAME)

        if show:
            plt.show()

    visualizeLearningHistory(history, show=True)

####################################################################################################
###> Load best model

if "evaluate" in TODO or "test" in TODO:

    print("\nLoading checkpoint %s" % (CHECKPOINTS_PATH + CHECKPOINTS_FILE_NAME))
    model.load_weights(CHECKPOINTS_PATH + CHECKPOINTS_FILE_NAME)

####################################################################################################
###> Evaluate model

if "evaluate" in TODO:

    def evaluate_model(model, x_test, y_test):
        
        print("\nEvaluating the model...")

        results = model.evaluate(
            x=x_test,
            y=y_test,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        loss = results[0]
        acc  = results[1]
        
        print("Test Loss: {:.5f}".format(loss))
        print("Test Accuracy: {:.2f}%".format(acc * 100))
        
    evaluate_model(model, x_test, y_test)

####################################################################################################
###> Test model

if "test" in TODO:

    def test(img, answer_coords=None, show=False, figure_name="Test"):

        print("\nTesting : " + str(figure_name))

        (img_w, img_h) = img.size[:2]
        img_array = keras.utils.img_to_array(img.resize(IMAGE_SIZE))
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        prediction_percentage_coords = model.predict(img_array)[0]
        prediction_percentage_boxes = [ 
            (
                int(prediction_percentage_coords[i+0] * img_w), 
                int(prediction_percentage_coords[i+1] * img_h), 
                int(prediction_percentage_coords[i+2] * img_w), 
                int(prediction_percentage_coords[i+3] * img_h)
            )
            for i in range(0, len(prediction_percentage_coords), 4)
        ]
        prediction_coords = [ coord for box in prediction_percentage_boxes for coord in box ]
        
        if not(answer_coords is None):
            print("Expected: " + str(answer_coords))
        print("Predicted: " + str(prediction_coords))

        displayImageWithTargets(img, answer_coords, prediction_coords, show=show, figure_name=figure_name)

    for image_id_to_test in IMAGE_IDS_TO_TEST:

        ( ( image, coords ), ( _, _ ) ) = extractDataFromId(image_id_to_test)
        test(image, answer_coords=coords, show=False, figure_name="Image " + str(image_id_to_test))
    
    plt.show()

####################################################################################################
###> Programm end message

print()
print("> Programm exited successfully!")
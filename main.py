import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os
import scipy.io
import absl.logging
import re
import random
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
DATASET_FOLDER     = "datasets/caltech-101/"
IMAGES_FOLDER      = DATASET_FOLDER + "101_ObjectCategories/airplanes/"
ANNOTATIONS_FOLDER = DATASET_FOLDER + "Annotations/Airplanes_Side_2/"
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

# TRAIN
TRAINING_PATIENCE   = 3
EPOCHS              = 250
BATCH_SIZE          = 16
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

        with open(fileimage, "rb") as fobj:
            if not tf.compat.as_bytes("JFIF") in fobj.peek(10):
                return True

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

def plotRectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y, color, linestyle=""):

    plt.plot(
        [ top_left_x, top_left_x, bottom_right_x, bottom_right_x, top_left_x ],
        [ top_left_y, bottom_right_y, bottom_right_y, top_left_y, top_left_y ], 
        color=color, linestyle=linestyle
    )

def displayImageWithTargets(img, tgts=None, preds=None, show=True, figure_name="Image with targets"):
    
    plt.figure(figure_name)
    plt.imshow(img)

    # Image dimentions
    image_w, image_h = image.size[:2]
    plotRectangle(0, 0, image_w, image_h, color="grey", linestyle="")

    # Targets
    if not(tgts is None):
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = tgts[3], tgts[1], tgts[2], tgts[0]
        plotRectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y, color="green", linestyle="dashed")

    # Pred
    if not(preds is None):
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = preds[3], preds[1], preds[2], preds[0]
        plotRectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y, color="orange", linestyle="dashed")

    if show:
        plt.show()

def getPathFromFolderAndId(folder, id):
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            currentId = list(map(int, re.findall(r'\d+', f)))[-1]
            if currentId == id:
                return full_path
    return None

def getPathsFromId(id):

    imagePath      = getPathFromFolderAndId(IMAGES_FOLDER, id)
    annotationPath = getPathFromFolderAndId(ANNOTATIONS_FOLDER, id)

    if not(imagePath) or not(annotationPath):
        print("\nWarning: Can't find the image with the id" + str(image_id_to_test) + " : test cancelled...")
        return None

    return imagePath, annotationPath

def extractDataFromId(id):

    image_path, annotation_path = getPathsFromId(id)

    image = keras.utils.load_img(image_path)
    box_coords  = tuple(scipy.io.loadmat(annotation_path)["box_coord"][0])
    
    return ( image, box_coords )

if "train" in TODO or "evaluate" in TODO:

    print("\nLoading the datasets...")

    images  = []
    targets = []
    ids = [ list(map(int, re.findall(r'\d+', f)))[-1]  for f in os.listdir(IMAGES_FOLDER)  if os.path.isfile(os.path.join(IMAGES_FOLDER, f)) ]
    random.shuffle(ids)

    for id in tqdm(ids):

        image, (top_left_x, top_left_y, bottom_right_x, bottom_right_y) = extractDataFromId(id)
        image_w, image_h = image.size[:2]

        # Resize image for ai processing
        images.append(keras.utils.img_to_array(image.resize(IMAGE_SIZE)))

        # Store position in %
        targets.append((
            float(top_left_x)        / image_w,
            float(top_left_y)        / image_h,
            float(bottom_right_x)    / image_w,
            float(bottom_right_y)    / image_h,
        ))

        # displayImageWithTargets(image, (top_left_x, top_left_y, bottom_right_x, bottom_right_y), figure_name="1", show=False)
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
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)

        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(DROPOUT)(x)
        outputs = layers.Dense(4, activation="relu")(x)

        return keras.Model(inputs, outputs)

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

    def test(img, answers=None, show=False, figure_name="Test"):

        print("\nTesting : " + str(figure_name))

        (img_w, img_h) = img.size[:2]
        img_array = keras.utils.img_to_array(img.resize(IMAGE_SIZE))
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        decimal_predictions = model.predict(img_array)[0]
        pred_top_left_x, pred_top_left_y          = int(decimal_predictions[0] * img_w), int(decimal_predictions[1] * img_h)
        pred_bottom_right_x, pred_bottom_right_y  = int(decimal_predictions[2] * img_w), int(decimal_predictions[3] * img_h)
        predictions = (pred_top_left_x, pred_top_left_y, pred_bottom_right_x, pred_bottom_right_y)
        
        if not(answers is None):
            print("Expected: " + str(answers))
        print("Predicted: " + str(predictions))

        displayImageWithTargets(img, answers, predictions, show=show, figure_name=figure_name)

    for image_id_to_test in IMAGE_IDS_TO_TEST:

        image, box = extractDataFromId(image_id_to_test)
        test(image, answers=box, show=False, figure_name="Image " + str(image_id_to_test))
    
    plt.show()

####################################################################################################
###> Programm end message

print()
print("> Programm exited successfully!")
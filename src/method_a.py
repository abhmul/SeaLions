import process_imgs as pmt

from keras.preprocessing.image import ImageDataGenerator

from models import inception_net as model_func

MODEL_NAME = model_func.__name__
BATCH_SIZE = 32
# TODO Change this depending on which model you run. Set up for InceptionV3
INPUT_SIZE = (299, 299)
# TODO Change this when running on the full dataset
train_ids = sld.trainshort_ids

# Loading the training data
sld = pmt.SeaLionData()
X = sld.load_train(train_ids, target_size=INPUT_SIZE)
# TODO Right now y is all the train counts, if you're using the trainshort,
# you want select those.
y = sld.pd_counts.values

# TODO Perform a validation split on the data. Use something like
# train_test_split in from scikit-learn

# Instantiate the model
model = model_func(INPUT_SIZE)

# This will save the best scoring model weights to the current directory
best_model_file = 'method_a_{}_weights.h5'.format(MODEL_NAME)
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min',
                             verbose=1, save_best_only=True,
                             save_weights_only=True)


# TODO Instantiate the train augmentor

# We don't want any image augmentation for the validation set
val_imgen = ImageDataGenerator(rescale = 1./255)

# TODO flow the generators from the data

# TODO Calculate the steps_per_epoch and validation_steps

# TODO use fit_generator to train the model

# TODO Run the model on the test data. You'll need to use predict_generator
# and sld.load_test_gen so you don't run out of memory

# TODO create a submission from the result of running the model. There's an
# auxilary for this in sld.

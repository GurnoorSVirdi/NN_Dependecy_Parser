from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras import layers
from keras.layers import Flatten, Embedding, Dense

def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    model = keras.Sequential()
    #the embeddiing layer
    model.add(layers.Embedding(input_dim=word_types, output_dim=32, input_length=6))
    #flatten the model
    model.add(Flatten())
    #dense of 100 units
    model.add(layers.Dense(100, activation="relu"))
    #dense of 10 units
    model.add(layers.Dense(10, activation="relu"))
    #output softmax layer
    model.add(layers.Dense(91, activation="softmax"))
    #adam optimizer
    model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")
    return model


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")
   
    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)
    
    model.save(sys.argv[3])

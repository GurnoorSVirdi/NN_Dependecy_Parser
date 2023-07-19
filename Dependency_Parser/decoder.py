from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from extract_training_data import FeatureExtractor, State


class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])
        # print(self.output_labels)

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)
        
        def check_action_validity(action, label, state):
            # this is a helper method to check if action is valid or not
            # must check the following items:
            # arc-left or right are not permitted if empty stack
            if len(state.stack) == 0 and (action == 'left_arc' or action == 'right_arc'):
                return False
            # Shifting the only word out of the buffer is also illegal, unless the stack is empty
            elif len(state.buffer) == 1 and action == 'shift':
                if len(state.stack) == 0:
                    return True
                return False
            # The root node must never be the target of a left-arc.
            # first check if its root node and with left arc
            elif (len(state.stack) == 1 and label == "root") and action == 'left_arc':
                #print(label)
                return False
            # otherwise valid
            else:
                return True

        while state.buffer:
            # TODO: Write the body of this loop for part 4
            #print("running_buffer", state.buffer)
            #print("running_stack", state.stack)
            # get the representation of the current state
            features = np.array(self.extractor.get_input_representation(words, pos, state))
            # was getting error similar to EDStem HW3 part 4 Evaluate Running Error  m1#635
            features = features.reshape(1, 6)

            # get the possible actions
            action_probabilities = self.model.predict(features)
            # now given the the probabilites for each action, we must get the possible actions using argsort
            # used HW 3 Part 4 Predict Output#621 for argsort
            actions = np.argsort(np.array(action_probabilities[0]))
            # reverse the array to get highest probabilities first
            possible_actions = actions[::-1]
            #print(possible_actions)

            # now we must go through the list of posssible actions and check validity
            for action in possible_actions:
                output_action, output_label = self.output_labels[action]
                #print("action", output_action)
                # print("label", output_label)

                if not check_action_validity(output_action, output_label, state):
                    # print("not valid action ran")
                    continue
                else:
                    # print("is valid", is_valid_action)
                    # print(output_label)
                    # if its valid check add label to right or left arc
                    if output_action == 'right_arc':
                        state.right_arc(output_label)
                        # print("ran right_arc")
                    elif output_action == 'left_arc':
                        state.left_arc(output_label)
                        # print("ran left_arc")
                    # otherwise it is a shift
                    else:
                        state.shift()
                    # once found valid action for this possible actions then we break ... found a legal transition
                    break

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], 'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

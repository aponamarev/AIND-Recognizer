import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_id in range(len(test_set.get_all_sequences())):

        sequence, lengths = test_set.get_item_Xlengths(word_id)

        world_LogLvalues = {}

        for word, model in models.items():

            try:
                world_LogLvalues[word] = model.score(sequence, lengths)
            except:
                world_LogLvalues[word] = float("-Infinity")

        probabilities.append(world_LogLvalues)
        guesses.append( max(world_LogLvalues, key=lambda x: world_LogLvalues[x]) )

    # return probabilities, guesses
    return (probabilities, guesses)

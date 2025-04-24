from allennlp.predictors.predictor import Predictor
import pandas as pd

# funciton takes in csv file path and returns it as a list of sentences with their events those events aspect along with the SRL data for
# that sentence, each instance is sentence, its SRL data, and a list of it's events with their aspect
def preprocess(filePath: str) -> list:
    # Load the SRL predictor
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    with open(filePath) as file:
        dataDF = pd.read_csv(file)
        sentences = [sent for sent in dataDF['Sentence'] if isinstance(sent, str)]
        # holds return data 
        data = []
        # holds the event and aspect data for each sentence 
        sentData = []
        # sentenceIndex is the index of each sentence in the sentence data
        sentenceIndex = -1
        # goes through each event in a sentence as it is paired with its aspect
        for info in zip(dataDF['Event'], dataDF['Aspect']):
            # skips first instance 
            if sentenceIndex != -1:
                # we check just the event to see if it not empty because the event and the aspect will match
                if isinstance(info[0], str): sentData.append(info)
                # only done when there are no more events from the data 
                else:
                    events = []
                    sentence = sentences[sentenceIndex]
                    predicted = predictor.predict(sentence=sentence) 
                    # goes through each verb as it is found in the predictor 
                    for verb in predicted['verbs']: 
                        event = {} 
                        # do we need?
                        event['sentence words'] = predicted['words']
                        # goes through each tag for each verb
                        for i, tag in enumerate(verb['tags']): 
                            # we ignore the first few characters as they are the BIO of the tag
                            # do we want to use BIO tag to make sure we get the start correct? do we need order
                            tag = tag[2:] if tag[2:] != 'V' else 'EVENT'
                            # can we order these just by index or would that ever create issues 
                            # this also will mark the different argm's as different things 
                            # do we need to worry about there being multiple types of a specific argm per event
                            if tag not in event: event[tag] = [i]
                            else: event[tag].append(i) 
                        events.append(event)
                    # [verb['tags'] for verb in predicted['verbs']]
                    # regex for [] regardless of what is in between if description
                    # do we need the sentence in here
                    data.append((sentence, events, sentData))
                    sentenceIndex += 1
                    sentData = []
            else: 
                sentenceIndex = 0
                continue
    return data 

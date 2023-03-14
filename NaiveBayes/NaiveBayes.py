# Character-based Naive Bayes classifier which classifies a document as English, Spanish, or
# Japanese. The documents only consist of 26 lower-case English alphabet characters and space.

import math

class NaiveBayesClassifier():
    def __init__(self):
        self.prior_pr_dict = dict()
        self.conditional_pr_dict = dict()
        self.log_pred_conditional_pr_dict = dict()
        self.log_pred_posterior_pr_dict = dict()
        self.acceptable_chars = []
        self.languages = []

    def train(self, data, characters, languages, alpha, K_L, K_S):
        """
        train function generates conditional and prior probabilities from the training set

        :param data: a list of dictionaries where each dictionary is a text file. The dictionary has
            two components:
                x: a list of all relevant characters in file, which are [a,b,c, ... ,,x,y,z,space]
                y: the file language label of English ('e'), Spanish ('s'), or Japanese ('j')
                numchars: integer denoting number of relevant characters in file
        :param characters: a char list of al relevant characters
        :param languages: a list of languages present in dataset in character form 
            (english -> 'e', spanish -> 's', japanese -> 'j')
        :param alpha: additive smoothing parameter
        :param K_L: number of possible labels (languages)
        :param K_S: number of possible characters
        :return: nothing
        """ 

        self.languages = languages
        self.acceptable_chars = characters

        # generate prior probabilities
        for language in languages:
            total = 0 #how many files of current langauge exist in data
            for i in range(0,len(data)):
                if data[i]['y'] == language:
                    total += 1
            self.prior_pr_dict[language] = (total+alpha)/(len(data)+(K_L*alpha))

        # generate conditional probabilities
        for language in languages:
            char_dict = dict()
            for char in characters:
                num_sum = 0 #sum in the numberator for eqtn shown in hw pdf
                denom_sum = 0 # sum in the denominator for eqtn show in hw pdf
                for i in range(0,len(data)):
                    if data[i]['y'] == language:
                        denom_sum += data[i]['numchars']
                        for j in range(0,data[i]['numchars']):
                            if data[i]['x'][j] == char:
                                num_sum += 1
                char_dict[char] = (num_sum+alpha)/(denom_sum+(K_S*alpha))
            self.conditional_pr_dict[language] = char_dict

    def predict(self, sample):
        """
        predict function generates a predicted language (label) for input sample
        
        :param sample: dictionary in bag of characters count form
        :return: prediction
        """
        
        # compute predicted conditional probabilities
        for language in self.languages:
            log_pred_conditional_pr = 0
            for char in self.acceptable_chars:
                log_pred_conditional_pr += (sample[char]*math.log(self.conditional_pr_dict[language][char]))
            self.log_pred_conditional_pr_dict[language] = log_pred_conditional_pr

        # compute predicted posterior probabilities
        for language in self.languages:
            self.log_pred_posterior_pr_dict[language] = self.log_pred_conditional_pr_dict[language] + math.log(self.prior_pr_dict[language])
        
        return max(self.log_pred_posterior_pr_dict, key=self.log_pred_posterior_pr_dict.get)

    def get_prior_pr_dict(self):
        return self.prior_pr_dict

    def get_conditional_pr_dict(self):
        return self.conditional_pr_dict

    def get_log_pred_conditional_pr_dict(self):
        return self.log_pred_conditional_pr_dict

    def get_log_pred_posterior_pr_dict(self):
        return self.log_pred_posterior_pr_dict

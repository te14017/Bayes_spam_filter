# Bayes_spam_filter
A spam email detection program based on Bayesian Probability

Bayesian Classifier for spam email detection.
 
This class implement the train method and isSpam method to train a classifier
and predict a given email.

      The training process works as follow:
      1. Learn the vocabulary of both spam and ham emails by extracting and cleaning terms from email text.
      2. Count and store the occurrence and frequency of terms into HashMap,
      3. Compute the spamicity P(Spam|Term) of each term according to their counted frequency, and stores it into HashMap.
 
      The predict process works as follow:
      1. Transform text into terms and clean them,
      2. Fetch the spamicities of terms from HashMap, and sort them by value of spamicities,
      3. Extract the number of considered terms according to their absolute spamicity compared to 0.5.
      4. Compute the probability of spam message and predict results according to the threshold.

#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification

    the list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    the actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project

    the data is stored in lists and packed away in pickle files at the end

"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        if True:
            #path = os.path.join('../../enron_mail_20110402/maildir/bailey-s/deleted_items/101')
            # [:-1] in path should remove the "." at the end but it does not,
            # so I add it again in the next line
            path = os.path.join('../..', path[:-1])
            print path
            email = open(path[:-1], "r")

            ### use parseOutText to extract the text from the opened email
            parsed = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            replace_words = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]
            for w in replace_words:
                parsed = parsed.replace(w, "")
            ### append the text to word_data
            word_data.append(parsed)
            ### a dictionary to help with the encoded from_data
            name_encoded = dict(zip(["sara", "chris"],[0, 1]))
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            from_data.append( name_encoded[name] )

            email.close()

print "-- emails processed --"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )


### in Part 4, do TfIdf vectorization here

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english')
vect.fit_transform(word_data)
print "#feature names: ", len(vect.get_feature_names())
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
# Whatever other imports you need

def vectorizer(dicio,dim):
        corpus =[] #all emails in one list
        authors = []
        for author,email_list in dicio.items():
            corpus += email_list
            authors += [author]*len(email_list)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        svd = TruncatedSVD(n_components=dim)
        reduced_features = svd.fit_transform(X)
        return reduced_features,authors

def create_dict(inputdir):
    dicio = defaultdict(list) #author:[txt1,txt2,...]
    for author in os.listdir(inputdir):
        author_dir = os.path.join(inputdir, author)
        #print(author_dir)
        for file in os.listdir(author_dir):
            file_path = os.path.join(author_dir, file)
            #print(file_path)
            with open(file_path, 'r') as f:
                text = f.read()
                dicio[author].append(text)
    return dicio
def data_cleaning(dicio):
        inner_dicio = dicio.copy()
        for author,email_list in inner_dicio.items():
            for index,email in enumerate(email_list):
                #print(email)
                #Excluir tudo antes de X-FileName, incluindo a sua linha
                splitted = email.split('\n')
                for i,line in enumerate(splitted):
                    if 'X-FileName' in line:
                        break
                splitted = splitted[i+1:] #Cleans the head of the email before the line with "X-File"
                for i, line in enumerate(splitted):
                    
                    if author[:-2] in line.lower():
                        break
                splitted = splitted[:i]
                for i,line in enumerate(splitted):
                    if '-----' in line:
                        break
                splitted = splitted [:i]

                # Cleans blank lines
                for line in splitted[::-1]:
                    if not any(line):
                        splitted.remove(line)

                inner_dicio[author][index] = '\n'.join(splitted)
        return inner_dicio
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.

    texts_dict = create_dict(args.inputdir)
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    data_cleaned = data_cleaning(texts_dict)
    X,y = vectorizer(data_cleaned,200)
    df = pd.DataFrame(X)
    df1 = pd.concat([df,pd.Series(y,name='label')],axis=1)

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    X_train, X_test, y_train, y_test = train_test_split(df1.iloc[:,:-1],df1.iloc[:,-1], test_size=args.testsize/100, random_state=42)
    train = pd.concat([X_train,y_train],axis=1)
    test = pd.concat([X_test,y_test],axis=1)
    train.to_csv('Train_corpus.csv')
    test.to_csv('Test_corpus.csv')

    print("Done!")
    

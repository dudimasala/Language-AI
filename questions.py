import nltk
import sys
import math
import os
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = {}
    for file in os.listdir(directory):
        fd = os.path.join(directory, file)
        with open(fd) as f:
            corpus[file] = f.read().replace('\n', '')
    
    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    allWords = nltk.word_tokenize(document.lower())
    stopWords = nltk.corpus.stopwords.words('english')
    punctuation = string.punctuation
    finalList = []
    for i in allWords:
        if i not in stopWords:
            wordList = list(i.split(" "))
            for char in range(len(wordList)):
                if wordList[char] in punctuation:
                    wordList[char] = ""
            
            newString = "".join(wordList)
            finalList.append(newString)
    
    return finalList
                
                    

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    finDict = {}
    for doc in documents.keys():
        words = documents[doc]
        nodup_words = list(dict.fromkeys(words))
        for word in nodup_words:
            if word in finDict.keys():
                finDict[word] += 1
            else:
                finDict[word] = 1
    
    for word in finDict.keys():
        finDict[word] = math.log(len(documents.keys())/finDict[word])
    
    return finDict
                


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    finalCounter = {}
    for file in files.keys():
        curSum = 0
        for word in query:
            tf = files[file].count(word)
            curSum += (tf * idfs[word])
        
        finalCounter[file] = curSum
        
    sort_fc = sorted(finalCounter.items(), key=lambda x: x[1], reverse=True)
    finalArr = []
    for file in sort_fc:
        finalArr.append(file[0])
    
    return finalArr[:n]
            
            
    


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    finalSentence = {}
    for sentence in sentences.keys():
        curVal = [0, 0]
        wordCount = 0
        for word in query:
            if word in sentences[sentence]:
                curVal[0] += idfs[word]
                wordCount += 1
        curVal[1] = wordCount / len(sentences[sentence])
        finalSentence[sentence] = curVal
    
    sort_fc = sorted(finalSentence.items(), key=lambda x: x[1], reverse=True)
    finalArr = []
    for file in sort_fc:
        finalArr.append(file[0])
    
    return finalArr[:n]    
              


if __name__ == "__main__":
    main()

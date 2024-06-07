'''
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from natsort import natsorted
import math
import numpy as np
import pandas as pd


file_name = natsorted(os.listdir('file'))


stemmer = PorterStemmer()


def preprocessing(doc):
    token_docs = word_tokenize(doc)
    prepared_doc = []

    for term in token_docs:
        
        stemmed_term = stemmer.stem(term.lower())
        prepared_doc.append(stemmed_term)

    return prepared_doc

document_of_terms = []
for file in file_name:
    with open(f'file/{file}', 'r') as f:
        document = f.read()
    document_of_terms.append(preprocessing(document))

print('Terms after tokenization and stemming ')
print(document_of_terms)



####### positional index #########
document_number = 0
positional_index = {}


for document in document_of_terms:

    # For position and term in the tokens.
    for positional, term in enumerate(document):
        # print(pos, '-->' ,term)
        
        # If term already exists in the positional index dictionary.
        if term in positional_index:
                
            # Increment total freq by 1.
            positional_index[term][0] = positional_index[term][0] + 1
                
            # Check if the term has existed in that DocID before.
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)
                    
            else:
                positional_index[term][1][document_number] = [positional]

        # If term does not exist in the positional index dictionary
        # (first encounter).
        else:
            
            # Initialize the list.
            positional_index[term] = []
            # The total frequency is 1.
            positional_index[term].append(1)
            # The postings list is initially empty.
            positional_index[term].append({})     
            # Add doc ID to postings list.
            positional_index[term][1][document_number] = [positional]

    # Increment the file no. counter for document ID mapping             
    document_number += 1

print('Positional index')
print(positional_index)

### phrase Query ###
query = input('Input Phrase Query: ')

def query_input(q):
    lis = [[] for _ in range(10)]
    for term in preprocessing(query):
        if term in positional_index:
            for key in positional_index[term][1].keys():
                if not lis[key - 1] or lis[key - 1][-1] == positional_index[term][1][key][0] - 1:
                    lis[key - 1].append(positional_index[term][1][key][0])

    positions = [f'doc{pos}' for pos, lst in enumerate(lis, start=1) if len(lst) == len(preprocessing(query))]
    return positions
print('++++++phrase query+++++')

print(query_input(query))

######### Print Tabels before input Query #############
all_words = []
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)

def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found

term_freq = pd.DataFrame(get_term_freq(document_of_terms[0]).values(), index=get_term_freq(document_of_terms[0]).keys())

for i in range(1, len(document_of_terms)):
    term_freq[i] = get_term_freq(document_of_terms[i]).values()

term_freq.columns = ['doc'+str(i) for i in range(1, 11)]
print('TF')
print(term_freq)

def get_weighted_term_freq(x):
    if x > 0:
        return math.log10(x) + 1
    return 0

for i in range(1, len(document_of_terms)+1):
    term_freq['doc'+str(i)] = term_freq['doc'+str(i)].apply(get_weighted_term_freq)

print('Weighted TF')
print(term_freq)
tfd = pd.DataFrame(columns=['freq', 'idf'])

for i in range(len(term_freq)):

    frequency = term_freq.iloc[i].values.sum()

    tfd.loc[i, 'freq'] = frequency

    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

tfd.index = term_freq.index

print('IDF')
print(tfd)

term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)

print('TF.IDF')
print(term_freq_inve_doc_freq)

import numpy as np
document_length = pd.DataFrame()

def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x**2).sum())

for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column+'_len'] = get_docs_length(column)

print('Document Length')
print(document_length)

normalized_term_freq_idf = pd.DataFrame()

def get_normalized(col, x):
    try:
        return x / document_length[col+'_len'].values[0]
    except:
        return 0

for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(lambda x : get_normalized(column, x))

print('Nomalized TF.IDF')
print(normalized_term_freq_idf)

######## input Query ##########

def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0

def insert_query(q):
    query = pd.DataFrame(index=normalized_term_freq_idf.index)
    query['tf'] = [1 if x in preprocessing(q) else 0 for x in list(normalized_term_freq_idf.index)]
    query['w_tf'] = query['tf'].apply(lambda x : get_w_tf(x))
    product = normalized_term_freq_idf.multiply(query['w_tf'], axis=0)
    query['idf'] = tfd['idf'] * query['w_tf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['normalized'] = 0
    for i in range(len(query)):
        query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))
    print('Query Details')
    print(query.loc[preprocessing(q)])
    product2 = product.multiply(query['normalized'], axis=0)
    scores = {}
    for col in product2.columns:
        if 0 in product2[col].loc[preprocessing(q)].values:
            pass
        else:
            scores[col] = product2[col].sum()
    product_result = product2[list(scores.keys())].loc[preprocessing(q)]
    print()
    print('Product (query*matched doc)')
    print(product_result)
    print()
    print('product sum')
    print(product_result.sum())
    print()
    print('Query Length')
    q_len = math.sqrt(sum([x**2 for x in query['idf'].loc[preprocessing(q)]]))
    print(q_len)
    print()
    print('Cosine Simliarity')
    print(product_result.sum())
    print()
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print('Returned docs')
    for typle in sorted_scores:
        print(typle[0])

q = input('Input Query for print Query details and matched document: ')
insert_query(q)

'''




'''

import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from natsort import natsorted
import math
import numpy as np
import pandas as pd

file_name = natsorted(os.listdir('file'))

stemmer = PorterStemmer()


def preprocessing(doc):
    token_docs = word_tokenize(doc)
    prepared_doc = []

    for term in token_docs:
        stemmed_term = stemmer.stem(term.lower())
        prepared_doc.append(stemmed_term)

    return prepared_doc


document_of_terms = []
for file in file_name:
    with open(f'file/{file}', 'r') as f:
        document = f.read()
    document_of_terms.append(preprocessing(document))

print('Terms after tokenization and stemming ')
print(document_of_terms)

####### positional index #########
document_number = 0
positional_index = {}

for document in document_of_terms:
    # For position and term in the tokens.
    for positional, term in enumerate(document):
        # If term already exists in the positional index dictionary.
        if term in positional_index:
            # Increment total freq by 1.
            positional_index[term][0] = positional_index[term][0] + 1

            # Check if the term has existed in that DocID before.
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)

            else:
                positional_index[term][1][document_number] = [positional]

        # If term does not exist in the positional index dictionary
        # (first encounter).
        else:
            # Initialize the list.
            positional_index[term] = []
            # The total frequency is 1.
            positional_index[term].append(1)
            # The postings list is initially empty.
            positional_index[term].append({})
            # Add doc ID to postings list.
            positional_index[term][1][document_number] = [positional]

    # Increment the file no. counter for document ID mapping
    document_number += 1

print('Positional index')
print(positional_index)

### phrase Query ###
query = input('Input Phrase Query: ')


def query_input(q):
    lis = [[] for _ in range(10)]
    for term in preprocessing(query):
        if term in positional_index:
            for key in positional_index[term][1].keys():
                if not lis[key - 1] or lis[key - 1][-1] == positional_index[term][1][key][0] - 1:
                    lis[key - 1].append(positional_index[term][1][key][0])

    positions = [f'doc{pos}' for pos, lst in enumerate(lis, start=1) if len(lst) == len(preprocessing(query))]
    return positions


print('++++++phrase query+++++')
print(query_input(query))

######### Print Tables before input Query #############
all_words = []
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)


def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(get_term_freq(document_of_terms[0]).values(), index=get_term_freq(document_of_terms[0]).keys())

for i in range(1, len(document_of_terms)):
    term_freq[i] = get_term_freq(document_of_terms[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print('TF')
print(term_freq)


def get_weighted_term_freq(x):
    if x > 0:
        return math.log10(x) + 1
    return 0


for i in range(1, len(document_of_terms) + 1):
    term_freq['doc' + str(i)] = term_freq['doc' + str(i)].apply(get_weighted_term_freq)

print('Weighted TF')
print(term_freq)
tfd = pd.DataFrame(columns=['freq', 'idf'])

for i in range(len(term_freq)):

    frequency = term_freq.iloc[i].values.sum()

    tfd.loc[i, 'freq'] = frequency

    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

tfd.index = term_freq.index

print('IDF')
print(tfd)

term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)

print('TF.IDF')
print(term_freq_inve_doc_freq)

import numpy as np

document_length = pd.DataFrame()


def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x ** 2).sum())


for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column + '_len'] = get_docs_length(column)

print('Document Length')
print(document_length)

normalized_term_freq_idf = pd.DataFrame()


def get_normalized(col, x):
    try:
        return x / document_length[col + '_len'].values[0]
    except:
        return 0


for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(lambda x: get_normalized(column, x))

print('Normalized TF.IDF')
print(normalized_term_freq_idf)


######## input Query ##########

def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0


def insert_query(q):
    query_terms = preprocessing(q)
    query = pd.DataFrame(index=normalized_term_freq_idf.index)
    query['tf'] = [1 if x in query_terms else 0 for x in list(normalized_term_freq_idf.index)]
    query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
    product = normalized_term_freq_idf.multiply(query['w_tf'], axis=0)
    query['idf'] = tfd['idf'] * query['w_tf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['normalized'] = 0

    for i in range(len(query)):
        query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values ** 2))

    print('Query Details')
    print(query.loc[query_terms])

    # Boolean operators
    and_docs = set(range(1, 11))  # Assume AND operation returns all documents initially
    or_docs = set()  # Assume OR operation returns an empty set initially

    for term in query_terms:
        if term in positional_index:
            and_docs &= set(positional_index[term][1].keys())
            or_docs |= set(positional_index[term][1].keys())

    and_docs = [f'doc{doc}' for doc in and_docs]
    or_docs = [f'doc{doc}' for doc in or_docs]

    print('AND Operation Result:')
    print(and_docs)

    print('OR Operation Result:')
    print(or_docs)

    product = normalized_term_freq_idf.multiply(query['normalized'], axis=0)

    scores_and = {}
    scores_or = {}

    for col in product.columns:
        if 0 in product[col].loc[query_terms].values:
            pass
        else:
            scores_and[col] = product[col].sum()

    for col in product.columns:
        scores_or[col] = product[col].sum()

    product_result_and = product[list(scores_and.keys())].loc[query_terms]
    product_result_or = product[list(scores_or.keys())].loc[query_terms]

    print()
    print('Product (query * matched doc) for AND Operation:')
    print(product_result_and)
    print()
    print('Product Sum for AND Operation:')
    print(product_result_and.sum())

    print()
    print('Product (query * matched doc) for OR Operation:')
    print(product_result_or)
    print()
    print('Product Sum for OR Operation:')
    print(product_result_or.sum())

    print()
    print('Query Length')
    q_len = math.sqrt(sum([x ** 2 for x in query['idf'].loc[query_terms]]))
    print(q_len)

    print()
    print('Cosine Similarity for AND Operation:')
    print(product_result_and.sum())

    print()
    print('Cosine Similarity for OR Operation:')
    print(product_result_or.sum())

    sorted_scores_and = sorted(scores_and.items(), key=lambda x: x[1], reverse=True)
    sorted_scores_or = sorted(scores_or.items(), key=lambda x: x[1], reverse=True)

    print('Returned docs for AND Operation:')
    for tuple in sorted_scores_and:
        print(tuple[0])

    print('Returned docs for OR Operation:')
    for tuple in sorted_scores_or:
        print(tuple[0])


q = input('Input Query for print Query details and matched document: ')
insert_query(q)




'''


'''

import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import math
import pandas as pd
import numpy as np
from natsort import natsorted
from sklearn.metrics.pairwise import cosine_similarity

file_name = natsorted(os.listdir('file'))
stemmer = PorterStemmer()


def preprocessing(doc):
    token_docs = word_tokenize(doc)
    prepared_doc = []

    for term in token_docs:
        stemmed_term = stemmer.stem(term.lower())
        prepared_doc.append(stemmed_term)

    return prepared_doc


document_of_terms = []
for file in file_name:
    with open(f'file/{file}', 'r') as f:
        document = f.read()
    document_of_terms.append(preprocessing(document))

print('Terms after tokenization and stemming ')
print(document_of_terms, "\n")

# Build positional index
document_number = 0
positional_index = {}

for document in document_of_terms:
    for positional, term in enumerate(document):
        if term in positional_index:
            positional_index[term][0] = positional_index[term][0] + 1
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)
            else:
                positional_index[term][1][document_number] = [positional]
        else:
            positional_index[term] = [1, {document_number: [positional]}]

    document_number += 1

# Display the positional index DataFrame
positional_index_df = pd.DataFrame.from_dict(positional_index, orient='index', columns=['Total Frequency', 'Postings List'])
print('\nPositional Index')
print(positional_index_df)

# Compute term frequency for each term in each document
all_words = [word for doc in document_of_terms for word in doc]

def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found

term_freq = pd.DataFrame(get_term_freq(document_of_terms[0]).values(), index=get_term_freq(document_of_terms[0]).keys())

for i in range(1, len(document_of_terms)):
    term_freq[i] = get_term_freq(document_of_terms[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print('\nTerm Frequency (TF)')
print(term_freq)

# Compute IDF for each term
tfd = pd.DataFrame(columns=['freq', 'idf'])

for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()
    tfd.loc[i, 'freq'] = frequency
    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

tfd.index = term_freq.index
print('\nInverse Document Frequency (IDF)')
print(tfd)

# Compute TF.IDF matrix for each term
term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)
print('\nTF.IDF Matrix')
print(term_freq_inve_doc_freq)

# Allow users to write a phrase query and display matched documents
# Allow users to write a phrase query and display matched documents
# Allow users to write a phrase query and display matched documents
# Allow users to write a phrase query and display matched documents
query = input('\nInput Phrase Query: ')

def query_input(q):
    lis = [[] for _ in range(10)]
    for term in preprocessing(q):
        if term in positional_index:
            for key in positional_index[term][1].keys():
                if not lis[key] or lis[key][-1] == positional_index[term][1][key][0] - 1:
                    lis[key].append(positional_index[term][1][key][0])

    positions = [f'doc{pos}' for pos, lst in enumerate(lis, start=1) if len(lst) == len(preprocessing(q))]
    return positions

print('\n++++++phrase query+++++')
matched_documents = query_input(query)
print(matched_documents)

# Compute similarity between the query and matched documents
query_terms = preprocessing(query)
query_vector = term_freq_inve_doc_freq.loc[query_terms, term_freq.columns].iloc[:, [int(doc[3:]) - 1 for doc in matched_documents]].values
matched_doc_vectors = term_freq_inve_doc_freq.loc[:, term_freq.columns].iloc[:, [int(doc[3:]) - 1 for doc in matched_documents]].values

# Ensure the number of columns is the same for query_vector and matched_doc_vectors
similarity_scores = cosine_similarity(query_vector, matched_doc_vectors)[0]

# Rank documents based on similarity score
ranked_docs = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)

# Display ranked documents
print('\nRanked Documents:')
for rank, (doc_index, similarity) in enumerate(ranked_docs, start=1):
    print(f"Rank {rank}: Document {doc_index + 1} - Similarity: {similarity:.4f}")

'''










#--------------------------------------------





import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import math
import pandas as pd
from natsort import natsorted
from sklearn.metrics.pairwise import cosine_similarity

file_name = natsorted(os.listdir('file'))
stemmer = PorterStemmer()

def preprocessing(doc):
    token_docs = word_tokenize(doc)
    prepared_doc = []

    for term in token_docs:
        stemmed_term = stemmer.stem(term.lower())
        prepared_doc.append(stemmed_term)

    return prepared_doc

document_of_terms = []
for file in file_name:
    with open(f'file/{file}', 'r') as f:
        document = f.read()
    document_of_terms.append(preprocessing(document))

print('Terms after tokenization and stemming ')
print(document_of_terms, "\n")

# Build positional index
document_number = 0
positional_index = {}

for document in document_of_terms:
    for positional, term in enumerate(document):
        if term in positional_index:
            positional_index[term][0] += 1
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)
            else:
                positional_index[term][1][document_number] = [positional]
        else:
            positional_index[term] = [1, {document_number: [positional]}]

    document_number += 1

# Display the positional index DataFrame
positional_index_df = pd.DataFrame.from_dict(positional_index, orient='index', columns=['Total Frequency', 'Postings List'])
print('\nPositional Index')
print(positional_index_df)

# Compute term frequency for each term in each document
all_words = [word for doc in document_of_terms for word in doc]

def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found

term_freq = pd.DataFrame(get_term_freq(document_of_terms[0]).values(), index=get_term_freq(document_of_terms[0]).keys())

for i in range(1, len(document_of_terms)):
    term_freq[i] = get_term_freq(document_of_terms[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print('\nTerm Frequency (TF)')
print(term_freq)

# Compute IDF for each term
tfd = pd.DataFrame(columns=['freq', 'idf'])

for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()
    tfd.loc[i, 'freq'] = frequency
    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

tfd.index = term_freq.index
print('\nInverse Document Frequency (IDF)')
print(tfd)

# Compute TF.IDF matrix for each term
term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)
print('\nTF.IDF Matrix')
print(term_freq_inve_doc_freq)

def evaluate_boolean_query(query_terms, boolean_operator):
    stack = []

    for term in query_terms:
        if term.lower() not in ["and", "or", "not"]:
            stack.append(query_input(term))
        elif term.lower() == "not":
            if len(stack) >= 1:
                operand = stack.pop()
                result = [i for i in range(1, 11) if i not in operand]
                stack.append(result)
            else:
                print("Error: Not enough operands for 'not' operator.")
                return []
        elif term.lower() == "and":
            if len(stack) >= 2:
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = list(set(operand1) & set(operand2))
                stack.append(result)
            else:
                print("Error: Not enough operands for 'and' operator.")
                return []
        elif term.lower() == "or":
            if len(stack) >= 2:
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = list(set(operand1) | set(operand2))
                stack.append(result)
            else:
                print("Error: Not enough operands for 'or' operator.")
                return []

    final_result = stack.pop() if stack else []
    print(f"Final Result: {final_result}")
    return final_result

# Function to handle phrase queries
def query_input(q):
    lis = [[] for _ in range(10)]
    query_terms = preprocessing(q)

    for term in query_terms:
        if term in positional_index:
            for key in positional_index[term][1].keys():
                if not lis[key] or lis[key][-1] == positional_index[term][1][key][0] - 1:
                    lis[key].append(positional_index[term][1][key][0])

    # Find common positions for all terms
    common_positions = set(lis[0]).intersection(*lis[1:])

    positions = [f'doc{pos}' for pos in common_positions]
    return positions

# Allow users to write a boolean or phrase query and display matched documents
query = input('\nInput Query: ')

# Check if the query is a boolean or phrase query
if any(op in query.lower() for op in ["and", "or", "not"]):
    boolean_query_terms = query.split()
    boolean_operator = boolean_query_terms[1] if len(boolean_query_terms) > 1 and boolean_query_terms[1].lower() in ["and", "or", "not"] else None
    matched_documents = evaluate_boolean_query(boolean_query_terms, boolean_operator)

    print('\n++++++boolean query+++++')
    print(f"Matched Documents: {matched_documents}")
else:
    matched_documents = query_input(query)
    print('\n++++++phrase query+++++')
    print(f"Matched Documents: {matched_documents}")

# Ensure there are matched documents before computing similarity scores
if not matched_documents:
    print("No documents matched the query. Exiting.")
    exit()

# Convert document names to indices
matched_documents_indices = [int(doc[3:]) - 1 for doc in matched_documents]

# Compute similarity between the query and matched documents
query_vector = term_freq_inve_doc_freq.loc[:, term_freq.columns].iloc[:, matched_documents_indices].values
matched_doc_vectors = term_freq_inve_doc_freq.iloc[:, matched_documents_indices].values

# Ensure the number of columns is the same for query_vector and matched_doc_vectors
similarity_scores = cosine_similarity(query_vector, matched_doc_vectors)[0]

# Rank documents based on similarity score
ranked_docs = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)

# Display ranked documents
print('\nRanked Documents:')
for rank, (doc_index, similarity) in enumerate(ranked_docs, start=1):
    print(f"Rank {rank}: Document {doc_index + 1} - Similarity: {similarity:.4f}")

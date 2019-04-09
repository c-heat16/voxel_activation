from nltk.corpus import brown, gutenberg, reuters
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gensim
import os
"""

    gutenberg: http://www.gutenberg.org/ 25,000 books 98,552 sents
    reuters: 10,788 news docs 54,716 sents
"""

model_file_tmplt = os.path.join('corpora', '{}.embedding')
pkl_file = 'fMRI_data/pkl/p1.pkl'
p_obj = pickle.load(open(pkl_file, 'rb'))
if not os.path.exists('corpora'):
    os.makedirs('corpora')

all_missing_words = []
all_words = []

for t, td in p_obj.info.items():
    word = td['word']
    all_words.append(word)

distinct_words, count = np.unique(all_words, return_counts=True)
distinct_words = list(distinct_words)
for w, c in zip(distinct_words, count):
    print('word: {} count: {}'.format(w, c))

for corp_name, corp_obj in zip(['Brown', 'Gutenberg', 'Reuters'], [brown, gutenberg, reuters]):
    model_file = model_file_tmplt.format(corp_name)
    if not os.path.exists(model_file):
        print('Training model for {} corpus...'.format(corp_name))
        model = gensim.models.Word2Vec(corp_obj.sents())
        model.save(model_file)
    else:
        print('Loading model for {} corpus...'.format(corp_name))
        model = gensim.models.Word2Vec.load(model_file)

    words_without_embedding = []
    word_embedding = dict()
    trial_embedding = dict()
    print('Getting embeddings...')
    for word in distinct_words:
        try:
            #print('Trial: {} word: {} embedding: {}'.format(t, td['word'], len(model[td['word']])))

            embedding = model[word]
            word_embedding[word] = embedding
            #trial_embedding[t] = embedding

        except Exception as ex:
            #print('Trial: {} word: {} No embedding'.format(t, td['word']))
            #input('Exception: {}'.format(ex))
            if word not in words_without_embedding:
                words_without_embedding.append(word)

    all_missing_words.append(words_without_embedding)

    #print('{} Words w/o embedding:'.format(corp_name))
    with open(os.path.join('corpora','{}_words_without_embedding.txt'.format(corp_name)), 'w+') as f:
        for word in words_without_embedding:
            f.write('{}\n'.format(word))
            #print('\t{}'.format(word))
    print('{} Num words w/o embedding: {}'.format(corp_name, len(words_without_embedding)))

    #with open(os.path.join('corpora', '{}_trial_embedding.pkl'.format(corp_name)), 'wb') as f:
    #    pickle.dump(trial_embedding, f, protocol=2)

    with open(os.path.join('corpora', '{}_word_embedding.pkl'.format(corp_name)), 'wb') as f:
        pickle.dump(word_embedding, f, protocol=2)

brown_missing_words = all_missing_words[0]
gutenberg_missing_words = all_missing_words[1]

print('Num unique words: {}'.format(len(distinct_words)))

common_missing_words = [w for w in brown_missing_words if w in gutenberg_missing_words]
with open(os.path.join('corpora','brown_gutenberg_words_without_embedding.txt'), 'w+') as f:
    for word in common_missing_words:
        f.write('{}\n'.format(word))

print('Num common missing words: {}'.format(len(common_missing_words)))

len_missing_words = [len(v) for v in all_missing_words]
len_missing_words.append(len(common_missing_words))
corpora_names = ['Brown', 'Gutenberg', 'Reuters', 'Brown & Gutenberg']
plt.bar(np.arange(len(len_missing_words)), len_missing_words)
plt.xticks(np.arange(len(len_missing_words)), corpora_names)
plt.title('# Missing Words Across Corpora')
plt.xlabel('Corpus')
plt.ylabel('# Missing Words')
plt.savefig('num_missing_words.pdf')

# import nltk
# nltk.download('wordnet')  
from nltk.corpus import wordnet as wn
word2idx=[]
idx2word=[]
def find_synonyms(word):
    '''use wordnet to find the synonyms'''
    synonyms = set()
    pos_acceptable={wn.NOUN:0,wn.ADJ:1,wn.ADJ_SAT:1,wn.ADV:0,wn.VERB:0} #for pos check
    #find the 同义词
    for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                pos=lemma.synset().name().split(".")[-2]
                if pos_acceptable[pos]: # POS check
                    synonyms.add(lemma.name())
    #find the 近义词
    for syn in wn.synsets(word):
        for syn_similar in syn.similar_tos():
            for lemma in syn_similar.lemmas():
                synonyms.add(lemma.name())
    return synonyms

def read_vocab():
    global word2idx, idx2word
    word2idx={}
    with open("/home/wrc/attack_on_prompt_learning/twitter_sentiment/vocab.txt","r",encoding='utf-8')as f:
        idx2word=f.readlines()
        for i in range(len(idx2word)):
            word=idx2word[i].strip()
            idx2word[i]=word
            word2idx[word]=i

def reparapgrasing(n_seed_words, p_seed_words):
    global word2idx,idx2word
    p_word_indices=[]
    n_word_indices=[]
    p_synonyms=set()
    n_synonyms=set()
    #loop through all the seed words and find their synonyms
    print("positive seed words")
    for word_idx in p_seed_words:
        word=idx2word[word_idx]
        print(word)
        if(len(word)<=3): #only accept the frequent word with len>3
            continue
        p_synonyms|=find_synonyms(word)
    print("negative seed words")
    for word_idx in n_seed_words:
        word=idx2word[word_idx]
        print(word)
        if(len(word)<=3): #only accept the frequent word with len>3
            continue
        n_synonyms|=find_synonyms(word)
    p_synonyms=list(p_synonyms)
    n_synonyms=list(n_synonyms)
    #loop
    for word in p_synonyms:
        try:
            p_word_indices.append(word2idx[word])
        except:
            pass
    for word in n_synonyms:
        try:
            n_word_indices.append(word2idx[word])
        except:
            pass
    return n_word_indices,p_word_indices
read_vocab()
        


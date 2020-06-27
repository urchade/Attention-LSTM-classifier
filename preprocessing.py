import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    def __init__(self, glove_path, data_path, sequence_length=70, test_val_size=3000):
        """
        Parameters
        ----------
        glove_path: str
            Path for glove vectors
        data_path: str
            Path for the data
        sequence_length: int
            Maximum lenght of sequence
        test_val_size: int
            Size of test and validation set
        """

        # Storing some variables
        self.seq_len = sequence_length
        self.data_path = data_path
        self.test_val_size = test_val_size
        self.pad_idx = 0  # putting padding index at 0

        print('Loading Spacy ....')
        # disable=['ner', 'tagger', 'parser'] for faster tokenization
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser'])

        print('Loading Glove ....')
        self.glove = dict()
        with open(glove_path, 'r', encoding='utf8') as f:
            for line in f:
                splits = line.split(' ')
                word = splits[0]
                vec = np.array(splits[1:], dtype=np.float32)
                self.glove[word] = vec
        # The dimension of the embedding is the size of a glove vector
        self.embedding_dim = vec.shape[0]
        print(f'\t- Number of vectors in Glove: {len(self.glove)}')
        print(f'\t- Emdedding dim: {self.embedding_dim}')

        # Splitting data to train, test and val
        print('Splitting data ...')
        self.train, self.test, self.val = self.get_data_splits()
        print(f'\t- Length of train set: {len(self.train)}\n'
              f'\t- Length of test set: {len(self.test)}\n'
              f'\t- Length of validation set: {len(self.val)}')

        self.n_classes = len(np.unique(self.train['label'].values))
        print(f'Number of classes: {self.n_classes}')

        print('Creating word dicts ....')
        # Creating word dictionary using the training set
        self.id2word, self.word2id = self.get_vocab_dicts()

        self.number_in_glove = 0  # Counter for number of word found in glove

        self.id2vec = {}  # a dictionary for word id/word vector pairs
        for (ids, word) in self.id2word.items():
            try:
                self.id2vec[ids] = self.glove[str(word)]
                self.number_in_glove += 1
            except KeyError:
                # Initialize the vector randomly If a word
                # in the dictionary is not found Glove
                self.id2vec[ids] = np.random.normal(size=self.embedding_dim)

        # Number of word in the dictionary
        self.num_words = len(self.id2vec)

        # Creating the embedding weight for initializing pytorch's embedding layer
        self.embedding_weight = np.array([vec for vec in self.id2vec.values()],
                                         dtype=np.float32)

        print(f"\t- Length of the vocabulary (training set): {self.num_words}\n"
              f"\t- Number of extracted vectors from Glove: {self.number_in_glove}\n"
              f"\t- Embedding weight shape: {self.embedding_weight.shape}")

    def get_vocab_dicts(self):
        """
        Returns
        -------
        Return word dictionaries using the training set
        """
        text_list = self.train['text'].apply(lambda x: [str(a.lemma_).lower() for a in self.nlp(x) if not a.is_punct])
        id2word = dict(enumerate(set([a for s in text_list for a in s]), start=1))
        id2word[0] = '<padding>'
        word2id = {ids: word for word, ids in id2word.items()}
        return id2word, word2id

    def get_data_splits(self):
        """
        Returns
        -------
        The randomly splitted data
        Same size for validation and testing
        """
        df = pd.read_csv(self.data_path)
        train, test_val = train_test_split(df, test_size=self.test_val_size, stratify=df['label'], random_state=1)
        test, val = train_test_split(test_val, test_size=int(self.test_val_size / 2), stratify=test_val['label'],
                                     random_state=1)
        train, test, val = train.reset_index(drop=True), test.reset_index(drop=True), val.reset_index(drop=True)
        return train, test, val

    def pad_sequence(self, list_ids):
        """
        Parameters
        ----------
        list_ids: A list of sequence of ids

        Returns
        -------
        Padded array
        """
        padded_array = np.ones(self.seq_len, dtype=np.int) * self.pad_idx
        for i, el in enumerate(list_ids[:self.seq_len]):
            padded_array[i] = el
        return padded_array

    def pad_mask(self, list_ids):
        """
        Parameters
        ----------
        list_ids: A list of padded sequence

        Returns
        -------
        Padding mask
        """
        return list(map(lambda a: 1 if a in [self.pad_idx] else 0, list_ids))

    def convert2idx(self, word_list):
        """
        Parameters
        ----------
        word_list: a list of tokens

        Returns
        -------
        A list of ids
        """
        lists = []
        for word in word_list:
            try:
                lists.append(self.word2id[word])
            except KeyError:
                # Just pass if the word is not in the dictionary (OOV)
                pass
        return lists

    def data2ids(self, data):
        """
        Apply tokenization, lemmatization, numericalization and padding.
        Parameters
        ----------
        data: pd.Series
        Returns
        -------
        text_ids, text_tokens
        """
        text_tokens = data.apply(lambda x: [str(a.lemma_).lower() for a in self.nlp(x) if not a.is_punct])
        text_ids = text_tokens.apply(lambda x: self.convert2idx(x))
        text_ids = text_ids.apply(lambda x: self.pad_sequence(x))
        pad_mask = text_ids.apply(lambda x: self.pad_mask(x))
        return text_ids, text_tokens, pad_mask

    def get_bert_ids(self, text, tokenizer):
        text = text.apply(lambda x: x.lower())
        ids = text.apply(lambda x: tokenizer.encode(x, max_length=self.seq_len, pad_to_max_length=True))
        mask = ids.apply(lambda x: list(map(lambda a: 0 if a in [tokenizer.pad_token_id] else 1, x)))
        return ids, mask

    def get_training_data(self, bert=False, tokenizer=None):
        """
        Returns
        -------
        The training data and its corresponding label
        """
        if bert:
            ids, mask = self.get_bert_ids(self.train['text'], tokenizer=tokenizer)
        else:
            ids, _, mask = self.data2ids(self.train['text'])

        return ids, mask, self.train['label']

    def get_validation_data(self, bert=False, tokenizer=None):
        """
        Returns
        -------
        The validation data and its corresponding label
        """
        if bert:
            ids, mask = self.get_bert_ids(self.val['text'], tokenizer=tokenizer)
        else:
            ids, _, mask = self.data2ids(self.val['text'])

        return ids, mask, self.val['label']

    def get_test_data(self, bert=False, tokenizer=None):
        """
        Returns
        -------
        The testing data and its corresponding label
        """
        if bert:
            ids, mask = self.get_bert_ids(self.test['text'], tokenizer=tokenizer)
        else:
            ids, _, mask = self.data2ids(self.test['text'])

        return ids, mask, self.test['label']

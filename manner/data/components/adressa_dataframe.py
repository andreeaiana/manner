from typing import Any, Dict, List, Tuple, Union

import os
import json
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from manner import utils
from manner.data.components.file_utils import (
    to_tsv,
    load_idx_map_as_dict,
    check_integrity
)
from manner.data.components.download_utils import (
    download_path,
    maybe_download
)
from manner.data.components.sentiment_annotator import SentimentAnnotator
from manner.data.components.adressa_user_info import UserInfo

tqdm.pandas()

log = utils.get_pylogger(__name__)


class AdressaDataFrame(Dataset):
    def __init__(
            self,
            seed: int,
            data_dir: str,
            adressa_url: str, 
            adressa_gzip_filename: str, 
            categ_embeddings_url: str,
            categ_embeddings_dirname: str,
            categ_embeddings_fpath: str,
            categ_embedding_dim: int,
            id2index_filenames: Dict[str, str],
            sentiment_model: str,
            tokenizer_max_length: int,
            train_day: List[int],
            test_day: List[int],
            neg_num: int,
            user_dev_size: float,
            train: bool,
            validation: bool,
            download: bool
            ) -> None:

        super().__init__()

        self.seed = seed
        self.data_dir = data_dir
        self.adressa_url = adressa_url
        self.adressa_gzip_filename = adressa_gzip_filename
        self.validation = validation 

        self.categ_embeddings_url = categ_embeddings_url
        self.categ_embeddings_dirname = categ_embeddings_dirname
        self.categ_embeddings_fpath = categ_embeddings_fpath
        self.categ_embedding_dim = categ_embedding_dim
        self.id2index_filenames = id2index_filenames

        self.sentiment_annotator = SentimentAnnotator(
                sentiment_model=sentiment_model, 
                tokenizer_max_length=tokenizer_max_length
                )
    
        self.train_day = train_day
        self.test_day = test_day
        self.neg_num = neg_num
        self.user_dev_size = user_dev_size

        if train:
            if not self.validation:
                self.data_split = "train"
            else:
                self.data_split = "dev"
        else:
            self.data_split = "test"

        self.dst_dir = os.path.join(self.data_dir, "Adressa_" + self.data_split)

        if download:
            self._download_dataset()
            self._download_and_extract_embeddings()

        if not check_integrity(os.path.join(self.data_dir, self.adressa_gzip_filename)):
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        self.news, self.behaviors = self.load_data()
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        user_bhv = self.behaviors.iloc[idx]

        history = user_bhv["history"]
        candidates = user_bhv["candidates"]
        labels = user_bhv["labels"]

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]
        labels = np.array(labels)

        return history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the parsed news and user behaviors.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                Tuple of news and behaviors datasets.
        """
        news = self._load_news()
        news = news[
            [
                "title",
                "category_label",
                "sentiment_label",
                "sentiment_score",
            ]
        ]
        log.info(f"News data size: {len(news)}")

        behaviors = self._load_behaviors()
        log.info(
            f"Behaviors data size for data split {self.data_split}, validation={self.validation}: {len(behaviors)}"
        )

        return news, behaviors

    def _load_news(self) -> pd.DataFrame:
        """Loads the parsed news. If not already parsed, loads and preprocesses the raw news data.

        Args:
            news (pd.DataFrame): Dataframe of news articles.

        Returns:
            pd.DataFrame: Parsed news data.
        """
        parsed_news_file = os.path.join(self.dst_dir, "parsed_news.tsv")

        if check_integrity(parsed_news_file):
            # news data already parsed
            log.info(f"News data already parsed. Loading from {parsed_news_file}.")
            news = pd.read_table(filepath_or_buffer=parsed_news_file)
            
        else:
            log.info(f"News data not parsed. Loading and preprocessing raw data.")

            raw_news_filepath = os.path.join(self.dst_dir, 'news.tsv')
            
            if not check_integrity(raw_news_filepath):
                log.info(f"Compressed files not processed. Reading news data.")
                news_title, news_category, news_subcategory, nid2index = self._process_news_files(os.path.join(self.data_dir, self.adressa_gzip_filename))
                self._write_news_files(news_title, news_category, news_subcategory, nid2index)

                news_title_df = pd.DataFrame(news_title.items(), columns=["id", "title"])
                to_tsv(news_title_df, os.path.join(self.dst_dir, "news_title.tsv"))

                nid2index_df = pd.DataFrame(nid2index.items(), columns=["id", "index"])
                to_tsv(nid2index_df, os.path.join(self.dst_dir, self.id2index_filenames["nid2index"]))


            log.info(f"Processing data.")
            columns_names = [
                "nid",
                "category",
                "subcategory",
                "title"
            ]
            news = pd.read_table(
                filepath_or_buffer=raw_news_filepath,
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
            )

            # compute sentiments
            log.info("Computing sentiments.")
            news['sentiment_predictions'] = news['title'].progress_apply(lambda title: self.sentiment_annotator(title))
            news['sentiment_class'], news['sentiment_score'] = zip(*news['sentiment_predictions'])
            news.drop(columns=['sentiment_predictions'], inplace=True)
            log.info("Sentiments computation completed.")
            
            # encode categorical aspects
            # categ2index map
            log.info('Constructing categ2index map.')
            news_category = news['category'].drop_duplicates().reset_index(drop=True)
            categ2index = {v: k+1 for k, v in news_category.to_dict().items()}
            for stage in ['train', 'dev', 'test']:
                dir_filepath = os.path.join(self.data_dir, "Adressa_" + stage)
                categ2index_fpath = os.path.join(dir_filepath, self.id2index_filenames['categ2index'])
                log.info(f'Saving categ2index map of size {len(categ2index)} in {categ2index_fpath}')
                to_tsv(df=pd.DataFrame(categ2index.items(), columns=['category', 'index']), 
                       fpath=categ2index_fpath)

            # sentiment2index map
            log.info('Constructing sentiment2index map.')
            news_sentiment = news['sentiment_class'].drop_duplicates().reset_index(drop=True)
            sentiment2index = {v: k+1 for k, v in news_sentiment.to_dict().items()}
            for stage in ['train', 'dev', 'test']:
                dir_filepath = os.path.join(self.data_dir, "Adressa_" + stage)
                sentiment2index_fpath = os.path.join(dir_filepath, self.id2index_filenames['sentiment2index'])
                log.info(f'Saving sentiment2index map of size {len(sentiment2index)} in {sentiment2index_fpath}')
                to_tsv(df=pd.DataFrame(sentiment2index.items(), columns=['sentiment', 'index']), 
                       fpath=sentiment2index_fpath)

            # generate category embeddings
            self._generate_category_embeddings(
                    word2index=categ2index,
                    embeddings_fpath=self.categ_embeddings_fpath,
                    embedding_dim=self.categ_embedding_dim,
                    transformed_embeddings_filename = 'categ_embedding'
                    )
            
            # encode categorical aspects
            news['category_label'] = news['category'].progress_apply(lambda x: categ2index.get(x, 0))
            news['sentiment_label'] = news['sentiment_class'].progress_apply(lambda x: sentiment2index.get(x, 0))

            # cache processed data
            for stage in ['train', 'dev', 'test']:
                parsed_news_filepath = os.path.join(self.data_dir, "Adressa_" + stage, "parsed_news.tsv")
                to_tsv(news, parsed_news_filepath)

        news = news.set_index("nid", drop=True)

        return news

    def _load_behaviors(self) -> pd.DataFrame:
        """Loads the parsed user behaviors. If not already parsed, loads and preprocesses the raw
        behavior data.

        Args:
            news (pd.DataFrame): Dataframe of news articles.

        Returns:
            pd.DataFrame: Parsed user behavior data.
        """
        parsed_behaviors_file = os.path.join(self.dst_dir, "parsed_behaviors_" + str(self.seed) + ".tsv")
        
        if check_integrity(parsed_behaviors_file):
            # beaviors data already parsed
            log.info(f"User behaviors data already parsed. Loading from {parsed_behaviors_file}.")
            behaviors = pd.read_table(
                filepath_or_buffer=parsed_behaviors_file,
                converters={
                    "history": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "candidates": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "labels": lambda x: list(map(int, x.strip("[]").split(", "))),
                },
            )
            behaviors["history"] = behaviors["history"].apply(lambda x: [int(e) for e in x])
            behaviors["candidates"] = behaviors["candidates"].apply(lambda x: [int(e) for e in x])
            
        else:
            log.info(f"User behaviors data not parsed. Loading and preprocessing raw data.")
            
            raw_behaviors_filepath = os.path.join(self.dst_dir, 'behaviors_' + str(self.seed) + ".tsv")

            if not check_integrity(raw_behaviors_filepath):
                news_title = load_idx_map_as_dict(os.path.join(self.dst_dir, "news_title.tsv"))
                nid2index = load_idx_map_as_dict(os.path.join(self.dst_dir, self.id2index_filenames["nid2index"]))

                log.info(f"Compressed files not processed. Reading behavior data.")
                uid2index, user_info = self._process_users(os.path.join(self.data_dir, self.adressa_gzip_filename), nid2index)

                log.info(f"Sorting user behavior data chronologically.")
                for uid in tqdm(user_info):
                    user_info[uid].sort_click()

                log.info(f"Constructing behaviors.")
                self.train_lines = []
                self.test_lines = []
                for uindex in tqdm(user_info):
                    uinfo = user_info[uindex]
                    train_news = uinfo.train_news
                    test_news = uinfo.test_news
                    hist_news = uinfo.hist_news
                    self._construct_behaviors(uindex, hist_news, train_news, test_news, news_title)

                shuffle(self.train_lines)
                shuffle(self.test_lines)

                test_split_lines, dev_split_lines = train_test_split(self.test_lines, test_size=self.user_dev_size, random_state=self.seed)    
                
                self._write_behavior_files(self.train_lines, "train")
                self._write_behavior_files(dev_split_lines, "dev")
                self._write_behavior_files(test_split_lines, "test")

            log.info(f"Compressed files read. Processing data.")
            columns_names = ["uid", "history", "impressions"]
            behaviors = pd.read_table(
                filepath_or_buffer=raw_behaviors_filepath,
                header=None,
                names=columns_names,
                usecols=range(len(columns_names)),
                low_memory=False
            )

            behaviors["history"] = behaviors["history"].fillna("").str.split()
            behaviors["history"] = behaviors["history"].apply(lambda x: [int(e) for e in x])
            behaviors["impressions"] = behaviors["impressions"].str.split()
            behaviors["candidates"] = behaviors["impressions"].apply(
                lambda x: [int(impression.split("-")[0]) for impression in x]
            )
            behaviors["labels"] = behaviors["impressions"].apply(
                lambda x: [int(impression.split("-")[1]) for impression in x]
            )
            behaviors = behaviors.drop(columns=["impressions"])

            # drop interactions of users without history
            count_interactions = len(behaviors)
            behaviors = behaviors[behaviors["history"].apply(len) > 0]
            dropped_interactions = count_interactions - len(behaviors)
            log.info(f"Removed {dropped_interactions} interactions without user history.")

            behaviors = behaviors.reset_index(drop=True)

            if self.data_split == "train":
                if not self.validation:
                    # compute uid2index map
                    log.info("Constructing uid2index map.")
                    uid2index = {}
                    for idx in tqdm(behaviors.index.tolist()):
                        uid = behaviors.loc[idx]["uid"]
                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index) + 1

                    fpath = os.path.join(self.dst_dir, self.id2index_filenames['uid2index'])
                    log.info(f"Saving uid2index map of size {len(uid2index)} in {fpath}")
                    to_tsv(df=pd.DataFrame(uid2index.items(), columns=["uid", "index"]), 
                           fpath=fpath)
                else:
                    # load uid2index map
                    log.info("Loading uid2index map.")
                    fpath = os.path.join(
                        self.data_dir, "Adressa_train", self.id2index_filenames['uid2index']
                    )
                    uid2index = load_idx_map_as_dict(fpath)

            else:
                # load uid2index map
                log.info("Loading uid2index map.")
                fpath = os.path.join(self.data_dir, "Adressa_train", self.id2index_filenames['uid2index'])
                uid2index = load_idx_map_as_dict(fpath)


            # map uid to index
            log.info("Mapping uid to index.")
            behaviors["user"] = behaviors["uid"].apply(lambda x: uid2index.get(x, 0))

            behaviors = behaviors[["user", "history", "candidates", "labels"]]

            # cache processed data
            log.info(
                f"Caching parsed behaviors of size {len(behaviors)} to {parsed_behaviors_file}."
            )
            to_tsv(behaviors, parsed_behaviors_file)

        return behaviors

    def _process_news_files(self, filepath) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, int]]:
        """
        Adapted from https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py
        """
        news_title = {}
        news_category = {}
        news_subcategory = {}
        
        tar = tarfile.open(filepath, "r:gz", encoding="utf-8")
        files = tar.getmembers()
        
        for file in tqdm(files):
            f = tar.extractfile(file)
            if f is not None:
                for line in f.readlines():
                    line = line.decode("utf-8")
                    event_dict = json.loads(line.strip('\n'))

                    if 'id' in event_dict and 'title' in event_dict and 'category1' in event_dict:
                        if event_dict['id'] not in news_title:
                            news_title[event_dict['id']] = event_dict['title']
                        else:
                            assert news_title[event_dict['id']] == event_dict['title']

                        if event_dict['id'] not in news_category:
                            news_category[event_dict['id']] = event_dict['category1'].split('|')[0]
                        else:
                            assert news_category[event_dict['id']] == event_dict['category1'].split('|')[0]
                        
                        if event_dict['id'] not in news_subcategory:
                            news_subcategory[event_dict['id']] = event_dict['category1'].split('|')[-1]
                        else:
                            assert news_subcategory[event_dict['id']] == event_dict['category1'].split('|')[-1]
            
        nid2index = {k: v for k, v in zip(news_title.keys(), range(1, len(news_title) + 1))}
        
        return news_title, news_category, news_subcategory, nid2index

    def _write_news_files(
            self, 
            news_title: Dict[str, str], 
            news_category: Dict[str, str], 
            news_subcategory: Dict[str, str], 
            nid2index: Dict[str, int] 
            ) -> None:
        """
        Adapted from https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py
        """
        
        news_lines = []
        for nid in tqdm(news_title):
            nindex = nid2index[nid]
            title = news_title[nid]
            category = news_category[nid] 
            subcategory = news_subcategory[nid]
            news_line = '\t'.join([str(nindex), category, subcategory, title]) + '\n'
            news_lines.append(news_line)

        for stage in ['train', 'dev', 'test']:
            filepath = os.path.join(self.data_dir, "Adressa_" + stage)
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
            with open(os.path.join(filepath, 'news.tsv'), 'w', encoding='utf-8') as f:
                f.writelines(news_lines)

    def _process_users(self, filepath: str, nid2index: Dict[str, int]) -> Tuple[Dict[str, int], Dict[int, Any]]:
        """
        Adapted from https://github.com/yjw1029/Efficient-FedRec/blob/839f967c1ed1c0cb0b1b4d670828437ffb712f29/preprocess/adressa_raw.py
        """
        uid2index = {}
        user_info = defaultdict(lambda: UserInfo(self.train_day, self.test_day))

        tar = tarfile.open(filepath, "r:gz", encoding="utf-8")
        files = tar.getmembers()
        
        for file in tqdm(files):
            f = tar.extractfile(file)
            if f is not None:
                for line in f.readlines():
                    line = line.decode("utf-8")
                    event_dict = json.loads(line.strip('\n'))

                    if 'id' in event_dict and 'title' in event_dict and event_dict['id'] in nid2index:
                        nindex = nid2index[event_dict['id']]
                        uid = event_dict['userId']

                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index)

                        user_index = uid2index[uid]
                        click_time = int(event_dict['time'])
                        day = int(file.name[-1]) 
                        user_info[user_index].update(nindex, click_time, day)

        return uid2index, user_info

    def _construct_behaviors(self, uindex, hist_news, train_news, test_news, news_title) -> None:
        probs = np.ones(len(news_title) + 1, dtype='float32')
        probs[hist_news] = 0
        probs[train_news] = 0
        probs[test_news] = 0
        probs[0] = 0
        probs /= probs.sum()

        train_hist_news = [str(i) for i in hist_news.tolist()]
        train_hist_line = " ".join(train_hist_news)

        for nindex in train_news:
            neg_cand = np.random.choice(
                   len(news_title) + 1, 
                   size=self.neg_num, 
                   replace=False, 
                   p=probs
                   ).tolist()
            cand_news = " ".join(
                   [f"{str(nindex)}-1"] + [f"{str(nindex)}-0" for nindex in neg_cand]
                   )
           
            train_behavior_line = f"{uindex}\t{train_hist_line}\t{cand_news}\n"
            self.train_lines.append(train_behavior_line) 

        test_hist_news = [str(i) for i in hist_news.tolist() + train_news.tolist()]
        test_hist_line = " ".join(test_hist_news)
        for nindex in test_news:
            neg_cand = np.random.choice(
                    len(news_title) + 1, 
                    size=self.neg_num, 
                    replace=False, 
                    p=probs
                    ).tolist()
            cand_news = " ".join(
                    [f"{str(nindex)}-1"] + [f"{str(nindex)}-0" for nindex in neg_cand]
                    )

            test_behavior_line = f"{uindex}\t{test_hist_line}\t{cand_news}\n"
            self.test_lines.append(test_behavior_line)

    def _write_behavior_files(self, behavior_lines, stage: str) -> None:
        filepath = os.path.join(self.data_dir, "Adressa_" + stage)
        with open(os.path.join(filepath, 'behaviors_' + str(self.seed) + ".tsv"), 'w', encoding='utf-8') as f:
            f.writelines(behavior_lines)

    def _generate_category_embeddings(self, word2index: Dict[str, int], embeddings_fpath: str, embedding_dim: int, transformed_embeddings_filename: Union[str, None]) -> None:
        """ Loads pretrained embeddings for the words in word_dict.

        Args:
            word2index (Dict[str, int]): word dictionary
            embeddings_fpath (str): the filepath of the embeddings to be loaded
            ebedding_dim (int): dimensionality of embeddings
            transformed_embeddings_filename (str): the name of the transformed embeddings file
        """

        embedding_matrix = np.random.normal(size=(len(word2index) + 1, embedding_dim))
        exist_word = set()

        with open(embeddings_fpath, "r") as f:
            for line in tqdm(f):
                linesplit = line.split(" ")
                word = line[0]
                if len(word) != 0:
                    if word in word2index:
                        embedding_matrix[word2index[word]] = np.asarray(list(map(float, linesplit[1:])))
                        exist_word.add(word)
        
        log.info(f'Rate of word missed in pretrained embedding: {(len(exist_word)/len(word2index))}.')

        for stage in ['train', 'dev', 'test']:
            fpath = os.path.join(self.data_dir, "Adressa_" + stage, transformed_embeddings_filename)
            if not check_integrity(fpath):
                log.info(f'Saving category embeddings in {fpath}')
                np.save(fpath, embedding_matrix, allow_pickle=True)

    def _download_dataset(self) -> None:
        """ Downloads the Adressa dataset, if not already downloaded."""

        log.info(f"Downloading Adressa dataset from {self.adressa_url} if not cached.")
        with download_path(self.data_dir) as path:
            path = maybe_download(url=self.adressa_url, filename=self.adressa_gzip_filename, work_directory=path)
            log.info(f"Compressed dataset downloaded.")
    
    def _download_and_extract_embeddings(self) -> None:
        """ Downloads and extracts Glove embeddings, if not already downloaded."""
        log.info(f"Downloading Glove embeddings from {self.categ_embeddings_url}.")

        glove_dst_dir = os.path.join(
                self.data_dir, self.categ_embeddings_dirname
                )

        # download the embeddings
        with download_path(self.data_dir) as path:
            path = maybe_download(
                    url=self.categ_embeddings_url,
                    filename=self.categ_embeddings_url.split('/')[-1],
                    work_directory=path
                    )
            log.info(f'Compressed Glove embeddings downloaded.')

            # extract the compressed file
            if not check_integrity(self.categ_embeddings_fpath):
                log.info(f'Extracting Glove embeddings from {path} intp {glove_dst_dir}.')
                tar = tarfile.open(os.path.join(self.data_dir, self.categ_embeddings_url.split('/')[-1]), 'r:gz')
                for member in tar.getmembers():
                    if member.isreg():
                        member.name = os.path.basename(member.name)
                        tar.extract(member, glove_dst_dir)
                tar.close()
                log.info(f'Embeddings extraction completed.')
    

import logging
from galvasr2.align.spark.align_lib import load_audio_id_text_id_mapping, load_transcripts
from datasketch import MinHash, MinHashLSH, MinHashLSHForest
from nltk import ngrams
from tqdm import tqdm
import numpy as np
import itertools

class DataDeduplication:
    
    def __init__(self, num_rows: int):
        self.num_rows = num_rows
    
    def read_transcription_data(
        self,
        spark: SparkSession,
        data_trans_index: str,
        data_trans: str,
    ) -> pd.DataFrame:
        # spark.sparkContext.setLogLevel("INFO") # "ALL" for very verbose logging
        logging.getLogger("py4j").setLevel(logging.ERROR)
        catalogue_df = load_audio_id_text_id_mapping(spark, data_trans_index)
        training_sample_rows = catalogue_df.collect()

        # Comment this out to load everything. It might takes ~15 minute, in my experience, on an 8 core machine.
        training_sample_rows = training_sample_rows[: self.num_rows]
        transcripts_df = load_transcripts(spark, data_trans, training_sample_rows)
        transcripts_pdf = transcripts_df.toPandas()
        return transcripts_pdf
    
    def generate_buckets(self, data:list, threshold:float=0.97) -> dict:
        # that accepts MinHash objects with 128 permutations functions
        lsh = MinHashLSH(threshold=0.97, num_perm=128)
        # Create MinHash objects
        minhashes = {}
        error = []
        for c, i in enumerate(tqdm(data)):
            try:
                if c%5000 == 0:
                    print(c)
                minhash = MinHash(num_perm=128)
                for d in ngrams(i, 3):
                    minhash.update("".join(d).encode('utf-8'))
                lsh.insert(c, minhash)
                minhashes[c] = minhash
            except:
                error.append(c)
                pass
        return lsh, minhashes
    
    def create_cand_pairs(self, lsh, minhashes):
        big_list = []
        for query in minhashes.keys():
            bucket = lsh.query(minhashes[query])
            if len(bucket)==1:
                big_list.append([bucket[0],"None"])
            if len(bucket)>1:
                first_val = bucket[0]
                for val in bucket[1:2]:
                    second_val = val
                    big_list.append([first_val,second_val])
        return big_list
    
    
    def find_duplicates(self, lsh, minhashes):
        duplicate = []
        for i in range(len(minhashes.keys())):
            try:
                result = lsh.query(minhashes[i])
                if len(result) > 1:
                    result.sort()
                    duplicate.append(result)
                    print((result))
            except:
                pass
        duplicate.sort()
        duplicate = list(duplicate for duplicate, _ in itertools.groupby(duplicate))
        return duplicate
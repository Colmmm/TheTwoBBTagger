from root_pandas import read_root
import pandas as pd
import gc
from tqdm import tqdm
from names_dict import ET_dict


train_path = '../TaggingJpsiK2012_fix_fix_5.root'
test_path = '../TaggingJpsiK2012_tiny_fix_fixSlice2.root'


ETdf_generator = read_root(paths=train_path, columns=self.ids+self.label+self.feats4MVA, flatten=self.flatfeats4MVA, chunksize=chunk_size)
        MVAdf = pd.DataFrame()
        for chunk_df in tqdm(MVAdf_generator, unit='chunks'):
            MVAdf = pd.concat([MVAdf, chunk_df])
            del chunk_df ; gc.collect()
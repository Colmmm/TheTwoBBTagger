from dataobjects import twoBBdf
from stages import firstStage, secondStage, thirdStage
from lof import LOF, combine
from names_dict import TB_dict, ET_dict

import gc ; gc.enable()
train_path = '../TaggingJpsiK2012_fix_fix_5_withLOF.root'
test_path = '../TaggingJpsiK2012_tiny_fix_fix_withLOF.root'

def main():
    # 1) TWO BODYS (TBs)
    trainTBs = twoBBdf(path=train_path, dict=TB_dict)
    testTBs = twoBBdf(path=test_path, dict=TB_dict)
    TB_train_df, TB_test_df, TB_train_scores, test_TB_scores = firstStage(train_TBs=trainTBs, test_TBs=testTBs, threshold=0.85, random_seed=42, chunk_size=200000)
    
    # 2) EXTRA TRACKS (ETs)
    trainETs = twoBBdf(path=train_path, dict=ET_dict, specific_TBs=TB_train_df.index)
    testETs = twoBBdf(path=test_path, dict=ET_dict, specific_TBs=TB_test_df.index)
    ET_train_df, ET_test_df = secondStage(train_ETs=trainETs, test_ETs=testETs, threshold=0.25, random_seed=42, chunk_size=100000)
    del trainTBs, testTBs, trainETs, testETs ; gc.collect()

    # 2.5) Combine TBs and ETs and apply LOF calculation
    TAG_train_df = combine(TB_COM_df=TB_train_df, ET_COM_df=ET_train_df); TAG_test_df = combine(TB_COM_df=TB_test_df, ET_COM_df=ET_test_df)
    del TB_train_df, TB_test_df, ET_train_df, ET_test_df ; gc.collect()
    #trainTAG_df.to_csv('trainTAG_df.csv'); testTAG_df.to_csv('testTAG_df.csv')

    # 3) LOF, combine TBs+ETs and then feed into tagger
    TAGs = thirdStage(train_TAG_df=TAG_train_df, test_TAG_df=TAG_test_df, train_TB_scores=TB_train_scores, test_TB_scores=test_TB_scores,
                      train_path=train_path, test_path= test_path, random_seed=42)

    return TAGs

if __name__ == '__main__':
    tags = main()
    tags.to_csv('FINAL_outputted_TAGS.csv')
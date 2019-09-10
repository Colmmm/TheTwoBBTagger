from dataobjects import twoBBdf
from stages import firstStage, secondStage, thirdStage
from lof import LOF, combine
from names_dict import TB_dict, ET_dict
train_path = '../TaggingJpsiK2012_tiny_fix_fix.root'
test_path = '../TaggingJpsiK2012_tiny_fix_fixSlice2.root'

def main():
    # 1) TWO BODYS (TBs)
    trainTBs = twoBBdf(path=train_path, dict=TB_dict)
    testTBs = twoBBdf(path=test_path, dict=TB_dict)
    trainpromisingTBs, testpromisingTBs = firstStage(train_TBs=trainTBs, test_TBs=testTBs, threshold=0.16, random_seed=42)
    
    # 2) EXTRA TRACKS (ETs)
    trainETs = twoBBdf(path=train_path, dict=ET_dict, specific_TBs=trainpromisingTBs.index)
    testETs = twoBBdf(path=test_path, dict=ET_dict, specific_TBs=testpromisingTBs.index)
    trainpromisingETs, testpromisingETs = secondStage(train_ETs=trainETs, test_ETs=testETs, threshold=0.2, random_seed=42)
    trainETs.specific_ETs = trainpromisingETs.index ; testETs.specific_ETs = testpromisingETs.index

    # 2.5) Combine TBs and ETs and apply LOF calculation
    trainTBs_df = trainTBs.get_MVAdf().loc[trainpromisingTBs.index] ; testTBs_df = testTBs.get_MVAdf().loc[testpromisingTBs.index]
    trainETs_df = trainETs.get_MVAdf().loc[trainpromisingETs.index]; testETs_df = testETs.get_MVAdf().loc[testpromisingETs.index]
    trainTAG_df = combine(TB_COM_df=trainTBs_df, ET_COM_df=trainETs_df); testTAG_df = combine(TB_COM_df=testTBs_df, ET_COM_df=testETs_df)
    #trainTAG_df.to_csv('trainTAG_df.csv'); testTAG_df.to_csv('testTAG_df.csv')

    # 3) LOF, combine TBs+ETs and then feed into tagger
    TAGs = thirdStage(train_TAG_df=trainTAG_df, test_TAG_df=testTAG_df, train_TB_scores=trainpromisingTBs, test_TB_scores=testpromisingTBs,
                      train_path=train_path, test_path= test_path, random_seed=42)
    return TAGs

if __name__ == '__main__':
    tags = main()

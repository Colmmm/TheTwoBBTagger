from dataobjects import twoBBdf
from stages import firstStage, secondStage, thirdStage
from lof import LOF, combine
from names_dict import TB_dict, ET_dict
path = '../TaggingJpsiK2012_tiny_fix_fix.root'


def main():
    TBs = twoBBdf(path=path, dict=TB_dict)
    promisingTBs = firstStage(TBs, threshold=0.2, random_seed=42)
    
    ETs = twoBBdf(path=path, dict=ET_dict, specific_TBs=promisingTBs.index )
    promisingETs = secondStage(ETs, threshold=0.5, random_seed=42) ; ETs.specific_ETs = promisingETs.index
    
    TAG_df = combine(TB_COM_df=LOF(TBs), ET_COM_df=LOF(ETs))
    TAGs = thirdStage(TAG_df, TB_scores=promisingTBs, path=path, random_seed=42)
    return TAGs


if __name__ == '__main__':
    tags = main()
def main():
    TBs = twoBBdf(path=path, dict=TB_dict)
    promisingTBs = firstStage(TBs, threshold=0.2, random_seed=42)
    
    ETs = twoBBdf(path=path, dict=ET_dict, specific_TBs=promisingTBs.index )
    promisingETs = secondStage(ETs, threshold=0.5, random_seed=42) ; ETs.specific_ETs = promisingETs.index
    
    TAG_df = combine(TB_COM_df=LOF(TBs), ET_COM_df=LOF(ETs))
    TAGs = thirdStage(TAG_df, TB_scores=promisingTBs, path=path, random_seed=42)
    return TAGs
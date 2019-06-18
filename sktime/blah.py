import pandas as pd

if __name__=="__main__":

    a0 = pd.Series(index=["a","aa","b","bb","ab"], data=[0,4,6,3,12])
    a1 = pd.Series(index=["bla","blah"], data=[31,2])

    b0 = pd.Series(index=["z","zz","zzz"], data=[3,5,8])
    b1 = pd.Series(index=["h","e","ll","o"], data=[6,6,6,8])

    df = pd.DataFrame()
    df['dim_0'] = [a0, b0]
    df['dim_1'] = [a1, b1]

    # first case, all dimensions (should be a0 and a1)
    print(df.iloc[0,:])
    print()

    # all cases, first dimensions (should be a0, b0)
    print(df.iloc[:,0])
    print()
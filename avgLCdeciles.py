dec = {name: pd.DataFrame() for name in df.columns}

for col in df.columns:
    test = df[~df[col].isna()] # removes blank space obs, ' '
#     test[col].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
    try:
        test[col] = test[col].str.replace(',', '').str.replace('$', '').astype(float)
    except:
        pass
#     test[col] = test[col].astype(float64)
#     test = test[~test[col].str.isalpha()]
    test = test[~test[col].astype(str).str.isalpha()]
    test[col] = test[col].astype(float) #Concerts from object
    try:
        qc = pd.qcut(test[col], q=10)
    except:
        qc = pd.qcut(test[col], q=10,duplicates = 'drop')
    t2 = test.copy()
    t2['RANGE'] = qc
    t2['count'] = 1
    t2['Average_LossCost'] = t2.XLOSSRATIO.astype(float)
    dec[col] = pd.DataFrame(t2[['count','RANGE', 'Average_LossCost']].groupby(by = ['RANGE']).sum())
    dec[col]=t2.groupby('RANGE').agg({'count':'sum','Average_LossCost':'mean'})
    dec[col]['Percent'] = dec[col]['count'] / dec[col]['count'].sum()
#     pd.concat([dec[col], t3])
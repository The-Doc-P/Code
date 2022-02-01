def rsample(dataframe, make, model, number):
    try:
#         make = list(make)
        filt = dataframe[dataframe['make'] == make]
        filt = filt[filt['model'].isin(model)]
        rsamp = filt.sample(n = number, random_state = 989)
#         print(rsamp.shape)
        return rsamp
    except:
        print('Not Enough Data here' + dataframe)
        
        
def faster(dataf, array):
#     model, P1,P2,P3,P4,P5,P6,P7,P8
    model = array[0]
    P1 = int(array[1])
    P2 = int(array[2])
    P3 = int(array[3])
    P4 = int(array[4])
    P5 = int(array[5])
    P6 = int(array[6])
    P7 = int(array[7])
    P8 = int(array[8])
    try:
        secondary = array[9]
    except:
        secondary = None
    try:
        tetriary = array[10]
    except:
        tetriary = None
    try:
        quaternary = array[11]
    except:
        quaternary = None
    try:
        order5 = array[12]
    except:
        order5 = None
    try:
        order6 = array[13]
    except:
        order6 = None
    try:
        order7 = array[14]
    except:
        order7 = None
    try:
        order8 = array[15]
    except:
        order8 = None
    try:
        order9 = array[16]
    except:
        order9 = None
    try:
        order10 = array[17]
    except:
        order10 = None
    try:
        order11 = array[18]
    except:
        order11 = None
    try:
        order12 = array[19]
    except:
        order12 = None
    try:
        order13 = array[20]
    except:
        order13 = None
    try:
        order14 = array[21]
    except:
        order14 = None
    try:
        order15 = array[22]
    except:
        order15 = None
    try:
        order16 = array[23]
    except:
        order16 = None
    try:
        dataf = dataf.append(rsample(d14,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P1))
    except:
        print('NOT ENOUGH DATA 2014 ' + masmake + ' ' +model)
        
    try:
        dataf = dataf.append(rsample(d15,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P2))
    except:
        print('NOT ENOUGH DATA 2015 ' + masmake + ' ' +model)
    try:
        dataf = dataf.append(rsample(d16,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P3))
    except:
        print('NOT ENOUGH DATA 2016 ' + masmake + ' ' +model)
    try:
        dataf = dataf.append(rsample(d17,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P4))
    except:
        print('NOT ENOUGH DATA 2017 ' + masmake + ' ' +model)
    try:
        dataf = dataf.append(rsample(d18,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P5))
    except:
        print('NOT ENOUGH DATA 2018 ' + masmake + ' ' +model)
    try:
        dataf = dataf.append(rsample(d19,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P6))
    except:
        print('NOT ENOUGH DATA 2019 ' + masmake + ' ' +model)
    try:
        dataf = dataf.append(rsample(d20,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P7))
    except:
        print('NOT ENOUGH DATA 2020 ' + masmake + ' ' +model)
    try:
        dataf = dataf.append(rsample(d21,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P8))
    except:
        print('NOT ENOUGH DATA 2021 ' + masmake + ' ' +model)
    return dataf


masmake2 = ['CADILLAC', 'CHEVROLET', 'CHRYSLER']


bigarray = [
#cadillac
[['ATS', '0', '0', '0', '0', '0', '0', '0', '0','ATS COUPE', 'ATS SEDAN', 'ATS-V COUPE', 'ATS-V SEDAN'],
['CT6', '0', '0', '0', '0', '0', '0', '0', '0','CT6 SEDAN', 'CT6-V'],
],
#Chevrolet
[[
['SILVERADO', '0', '0', '0', '0', '0',
 '0', '0', '0', 'SILVERADO 1500', 'SILVERADO 1500 LD','SILVERADO MD'],
['SILVERADO 2500 3500', '0', '0', '0', '0',
 '0', '0', '0', '0','3500 GAS', '3500 LCF GAS', '3500HD DIESEL',
 'SILVERADO 2500HD','SILVERADO 2500HD BUILT AFTER AUG 14', 'SILVERADO 3500HD',
 'SILVERADO 3500HD BUILT AFTER AUG 14', 'SILVERADO 3500HD CC','2_3_series'],
],
 # Chrysler
[
['PACIFICA', '0', '0', '0', '0', '0', '0', '0',
 '0','PACIFICA HYBRID'],
['TOWN & COUNTRY', '14216', '10392', '7240', '0', '0', '0', '0',
 '0','TOWN &AMP; COUNTRY'],
['VOYAGER', '0', '0', '0', '0', '0', '0', '0', '0']]
]

rest3 = pd.DataFrame()
for masmake, barray in zip(masmake2,bigarray):
    for array in barray:
        rest3 = faster(rest3, array)
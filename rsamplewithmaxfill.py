def rsample(dataframe, make, model, number):
    try:
#         make = list(make)
        filt = dataframe[dataframe['make'] == make]
        filt = filt[filt['model'].isin(model)]
        try:
            rsamp = filt.sample(n = number, random_state = 727)
        except:
            rsamp = filt
#         print(rsamp.shape)
        return rsamp
    except:
        print('Not Enough Data here' + dataframe)
        
        
# V3

# Need to add several more orders to accomodate BMW

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
        yr_holder = d14
        PX = yr_holder[yr_holder['model'] == model].shape[0]
        print('NOT ENOUGH DATA 2014 ' + masmake + ' ' +model + ' New pull number = ' + str(PX))
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    try:
        dataf = dataf.append(rsample(d15,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P2))
    except:
        yr_holder = d15
        try:
            PX = yr_holder[yr_holder['model'] == model].shape[0]
        except:
            PX = 0
        print('NOT ENOUGH DATA 2015 ' + masmake + ' ' +model + ' New pull number = ' + str(PX))
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    try:
        dataf = dataf.append(rsample(d16,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P3))
    except:
        yr_holder = d16
        PX = yr_holder[yr_holder['model'] == model].shape[0]
        print('NOT ENOUGH DATA 2016 ' + masmake + ' ' +model + ' New pull number = ' + str(PX))
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    try:
        dataf = dataf.append(rsample(d17,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P4))
    except:
        yr_holder = d17
        PX = yr_holder[yr_holder['model'] == model].shape[0]
        print('NOT ENOUGH DATA 2017 ' + masmake + ' ' +model + ' New pull number = ' + str(PX))
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    try:
        dataf = dataf.append(rsample(d18,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P5))
    except:
        yr_holder = d18
        PX = yr_holder[yr_holder['model'] == model].shape[0]
        print('NOT ENOUGH DATA 2018 ' + masmake + ' ' +model + ' New pull number = ' + str(PX))
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    try:
        dataf = dataf.append(rsample(d19,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P6))
    except:
        yr_holder = d19
        PX = yr_holder[yr_holder['model'] == model].shape[0]
        print('NOT ENOUGH DATA 2019 ' + masmake + ' ' +model + ' New pull number = ' + str(PX))
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    try:
        dataf = dataf.append(rsample(d20,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P7))
    except:
        yr_holder = d20
        PX = yr_holder[yr_holder['model'] == model].shape[0]
        print('NOT ENOUGH DATA 2020 ' + masmake + ' ' +model + ' New pull number = ' + str(PX))
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    try:
        dataf = dataf.append(rsample(d21,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],P8))
    except:
        yr_holder = d21
        PX = yr_holder[yr_holder['model'] == model].shape[0]
        print('NOT ENOUGH DATA 2021 ' + masmake + ' ' +model + ' New pull number = ' + str(PX) )
        dataf = dataf.append(rsample(yr_holder,masmake,[model,secondary,tetriary,quaternary,order5,order6,order7,order8,order9,order10,order11,order12,order13,order14,order15,order16],PX))
    return dataf
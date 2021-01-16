import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import statistics
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

def zad1():
    con = pd.DataFrame(pd.read_csv("time_series_covid19_confirmed_global.csv"))
    rec = pd.DataFrame(pd.read_csv("time_series_covid19_recovered_global.csv"))
    death = pd.DataFrame(pd.read_csv("time_series_covid19_deaths_global.csv"))

    l = list(con.columns)[4:]
    con = pd.concat([con.groupby(['Country/Region']).agg({'Lat':'mean', 'Long':'mean'}), con.groupby(['Country/Region'])[l].sum()], axis=1)
    rec = pd.concat([rec.groupby(['Country/Region']).agg({'Lat':'mean', 'Long':'mean'}), rec.groupby(['Country/Region'])[l].sum()], axis=1)
    death = pd.concat([death.groupby(['Country/Region']).agg({'Lat':'mean', 'Long':'mean'}), death.groupby(['Country/Region'])[l].sum()], axis=1)

    nonrec = []
    nrec = rec.iloc[:, 2:].sum(axis=1)
    for ind, country in enumerate(list(nrec.index)):
        if nrec.iloc[ind] == 0:
            nonrec.append(ind)
    for country in nonrec:
        for ind, day in enumerate(list(con.columns)):
            if ind == (len(list(con.columns)) - 16):
                break
            else:
                rec.iloc[country, ind + 16] = con.iloc[country, ind + 2] - death.iloc[country, ind + 16]

    ndea = death.iloc[:, 2:].sum(axis=1)
    nondea = []
    for country in list(ndea.index):
        if ndea.loc[country] == 0:
            nondea.append(country)
    for country in nondea:
        rec.drop(country, inplace=True)
        con.drop(country, inplace=True)
        death.drop(country, inplace=True)
    #print(con)

    result = con.iloc[:, 2:] - rec.iloc[:, 2:] - death.iloc[:, 2:]
    #print("Aktywne przypadki:\n", result)
    death_m = death.iloc[:, 2:]
    rec_m = rec.iloc[:, 2:]
    death_m.columns = pd.to_datetime(death_m.columns).to_period('M')
    rec_m.columns = pd.to_datetime(rec_m.columns).to_period('M')
    death_m = death_m.groupby(death_m.columns, axis=1).sum()
    rec_m = rec_m.groupby(rec_m.columns, axis=1).sum()
    result2 = death_m/rec_m
    # #print("Skumulowana śmiertelność:\n", result2)
    return death, rec, con, result


def zad2(con, active):
    col = list(active.columns)
    rep = pd.DataFrame(0, index=active.index, columns=col[6:])
    for ind, c in enumerate(col):
        if ind == len(col) - 6:
            break
        val = pd.DataFrame(0, index=active.index, columns=['Value'])
        for country in range(len(active.index)):
            numdays = 0
            for i in range(7):
                if active.iloc[country, ind + i] >= 100:
                    val.iloc[country, 0] = val.iloc[country, 0] + active.iloc[country, ind + i]
                    numdays += 1
            if numdays == 0:
                rep.iloc[country, ind] = np.nan
            else:
                rep.iloc[country, ind] = val.iloc[country, 0]/numdays

    rep_col = list(rep.columns)
    result3 = pd.DataFrame(0, index=active.index, columns=rep_col[5:])
    for ind, c in enumerate(rep_col):
        if ind == len(rep_col) - 5:
            break
        result3.iloc[:, ind] = rep.iloc[:, ind + 5]/rep.iloc[:, ind]
    result3['Lat'] = con['Lat']
    result3['Long'] = con['Long']
    cols = list(result3.columns)
    cols = cols[-2:] + cols[:-2]
    result3 = result3[cols]
    return result3

def zad3():
    weatherM = Dataset('TerraClimate_tmax_2018.nc')
    weatherm = Dataset('TerraClimate_tmin_2018.nc')
    meanT = []

    for i in range(0, 12):
        sm = pd.DataFrame(weatherm['tmin'][i])
        sM = pd.DataFrame(weatherM['tmax'][i])
        meanT.append((sm + sM) / 2)
    ind = []
    col = []
    for i in range(4320):
        ind.append(-(i/24 - 90))
    for i in range(8640):
        col.append(i/24 - 180)
    for i in range(12):
        meanT[i].index = ind
        meanT[i].columns = col

    return meanT


def zad4(rep, meanT):
    rep_m = rep.iloc[:, 2:]
    rep_m.columns = pd.to_datetime(rep_m.columns).month
    rep_m = rep_m.groupby(rep_m.columns, axis=1).mean()
    for ind, country in enumerate(list(rep_m.index)):
        m = max(rep_m.iloc[ind, :].fillna(0))
        rep_m.iloc[ind, :] = rep_m.iloc[ind, :]/m

    print(rep_m)
    # słownik {<0 : 0, 0-10 : 1, 10-20: 2, 20-30 : 3, 30> : 4}
    for i in range(len(meanT)):
        print(i)
        for col in list(meanT[i].columns):
            meanT[i].loc[meanT[i][col] < 0, col] = 0
            meanT[i].loc[(meanT[i][col] >= 0) & (meanT[i][col] < 10), col] = 1
            meanT[i].loc[(meanT[i][col] >= 10) & (meanT[i][col] < 20), col] = 2
            meanT[i].loc[(meanT[i][col] >= 20) & (meanT[i][col] < 30), col] = 3
            meanT[i].loc[meanT[i][col] >= 30, col] = 4

        meanT[i].to_csv("temperature" + str(i) + ".csv")
   # print(meanT[0])

    for i in range(len(meanT)):
        zero=[]
        one=[]
        two=[]
        three=[]
        four=[]
        lists = [zero, one, two, three, four]
        n = ['zero', 'one', 'two', 'three', 'four']
        current=[]
        names=[]
        for ind, country in enumerate(list(rep.index)):
            lat = rep.loc[country, 'Lat']
            long = rep.loc[country, 'Long']
            latT = min(list(meanT[0].index), key=lambda x: abs(x - lat))
            longT = min(list(meanT[0].columns), key=lambda x: abs(x - long))
            temp = meanT[i].loc[latT, longT]
            if temp == np.nan or pd.isnull(rep_m[i+1][country]):
                continue
            elif temp == 0:
                zero.append(rep_m[i+1][country])
            elif temp == 1:
                one.append(rep_m[i+1][country])
            elif temp == 2:
                two.append(rep_m[i+1][country])
            elif temp == 3:
                three.append(rep_m[i+1][country])
            elif temp == 4:
                four.append(rep_m[i+1][country])
        for ind, lis in enumerate(lists):
            if len(lis) != 0:
                current.append(lis)
                names.append(n[ind])

        if len(current) > 1:
            f_value, p_value = f_oneway(*current)
            if p_value < 0.05:
                print(pairwise_tukeyhsd(np.concatenate(current), np.concatenate(
                    [[names[ind]]*len(c) for ind, c in enumerate(current)])))

            print(f'F-stat: {f_value}, p-val: {p_value}')


if __name__ == "__main__":
    death, rec, con, active = zad1()
    rep = zad2(con, active)
    mean = zad3()
    zad4(rep, mean)

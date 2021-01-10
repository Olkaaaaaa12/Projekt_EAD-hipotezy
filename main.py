import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import statistics
from scipy import stats

def zad1():
    con = pd.DataFrame(pd.read_csv("time_series_covid19_confirmed_global.csv"))
    rec = pd.DataFrame(pd.read_csv("time_series_covid19_recovered_global.csv"))
    death = pd.DataFrame(pd.read_csv("time_series_covid19_deaths_global.csv"))

    l = list(con.columns)[4:]
    con = pd.concat([con.groupby(['Country/Region']).agg({'Lat':'mean', 'Long':'mean'}), con.groupby(['Country/Region'])[l].sum()], axis=1)
    rec = pd.concat([rec.groupby(['Country/Region']).agg({'Lat':'mean', 'Long':'mean'}), rec.groupby(['Country/Region'])[l].sum()], axis=1)
    death = pd.concat([death.groupby(['Country/Region']).agg({'Lat':'mean', 'Long':'mean'}), death.groupby(['Country/Region'])[l].sum()], axis=1)
    result = con.iloc[:, 2:] - rec.iloc[:, 2:] - death.iloc[:, 2:]
    #print("Aktywne przypadki:\n", result)

    death_m = death.iloc[:, 2:]
    rec_m = rec.iloc[:, 2:]
    death_m.columns = pd.to_datetime(death_m.columns).to_period('M')
    rec_m.columns = pd.to_datetime(rec_m.columns).to_period('M')
    death_m = death_m.groupby(death_m.columns, axis=1).sum()
    rec_m = rec_m.groupby(rec_m.columns, axis=1).sum()
    result2 = death_m/rec_m
    #print("Skumulowana śmiertelność:\n", result2)
    return death, rec, con


def zad2(con):
    col = list(con.columns)[2:]
    rep = pd.DataFrame(0, index=con.index, columns=col[6:])
    for ind, c in enumerate(col):
        if ind == len(col) - 6:
            break
        val = pd.DataFrame(0, index=con.index, columns=['Value'])
        for i in range(7):
            val['Value'] = val['Value'] + con.iloc[:, ind + 2 + i]
        rep[col[ind + 6]] = val['Value']/7
    rep_col = list(rep.columns)
    result3 = pd.DataFrame(0, index=con.index, columns=rep_col[5:])
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
    mean = []

    for i in range(0, 12):
        sm = pd.DataFrame(weatherm['tmin'][i])
        sM = pd.DataFrame(weatherM['tmax'][i])
        mean.append((sm + sM) / 2)
    ind = []
    col = []
    for i in range(4320):
        ind.append(-(i/24 - 90))
    for i in range(8640):
        col.append(i/24 - 180)
    for i in range(12):
        mean[i].index = ind
        mean[i].columns = col
    mean.pop(0)
    return mean

def zad4(rep, meanD):
    rep = rep.replace(np.inf, np.nan)
    rep = rep.fillna(0)

    for ind, country in enumerate(list(rep.index)):
        m = max(rep.iloc[ind, 2:])
        rep.iloc[ind, 2:] = rep.iloc[ind, 2:]/m
    rep_m = rep.iloc[:, 2:]
    rep_m.columns = pd.to_datetime(rep_m.columns).to_period('M')
    rep_m = rep_m.groupby(rep_m.columns, axis=1).mean()
    list_nan = []
    for ind, country in enumerate(list(rep.index)):
        lat = rep.loc[country, 'Lat']
        long = rep.loc[country, 'Long']
        latT = min(list(meanD[0].index), key=lambda x: abs(x - lat))
        longT = min(list(meanD[0].columns), key=lambda x: abs(x - long))
        temp = []
        for i in range(len(meanD)):
           temp.append(meanD[i].loc[latT, longT])

        _, p = stats.normaltest(rep_m.iloc[ind, :])
        _, p2 = stats.normaltest(temp)

        if np.isnan(p2):
            list_nan.append(country)
        if p > 0.95 or p2 > 0.95:
            print(country)
    for c in list_nan:
        rep_m.drop([c], inplace=True)
    print(rep_m)

if __name__ == "__main__":
    death, rec, con = zad1()
    rep = zad2(con)
    mean = zad3()
    zad4(rep, mean)

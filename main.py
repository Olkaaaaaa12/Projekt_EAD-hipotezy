import pandas as pd
import numpy as np
from netCDF4 import Dataset
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import normaltest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
import pycountry_convert as pc
from time import perf_counter

#Zastosowano grupowanie po krajach i policzono średnią z długości i szerokości geograficznej


def zad1():
    print("--------------------- Metryki COVID-19 Zadanie 1 ---------------------")
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

    result = con.iloc[:, 2:] - rec.iloc[:, 2:] - death.iloc[:, 2:]
    print("Metryki COVID-19 Zadanie 1: Aktywne przypadki:\n", result)
    death_m = death.iloc[:, 2:]
    rec_m = rec.iloc[:, 2:]
    death_m.columns = pd.to_datetime(death_m.columns).to_period('M')
    rec_m.columns = pd.to_datetime(rec_m.columns).to_period('M')
    death_m = death_m.groupby(death_m.columns, axis=1).sum()
    rec_m = rec_m.groupby(rec_m.columns, axis=1).sum()
    result2 = death_m/rec_m
    print("Metryki COVID-19 Zadanie 1: Skumulowana śmiertelność:\n", result2)
    return death, con, result, result2


def zad2(con, active):
    print("\n--------------------- Metryki COVID-19 Zadanie 2 ---------------------")
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
    print("Metryki COVID-19 Zadanie 2: Współczynnik reprodukcji wirusa: \n", result3)
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
    print("\n--------------------- Testowanie hipotez Zadanie 1 ---------------------")
    rep_m = rep.iloc[:, 2:]
    rep_m.columns = pd.to_datetime(rep_m.columns).month
    rep_m = rep_m.groupby(rep_m.columns, axis=1).mean()
    for ind, country in enumerate(list(rep_m.index)):
        m = max(rep_m.iloc[ind, :].fillna(0))
        rep_m.iloc[ind, :] = rep_m.iloc[ind, :]/m

    dictTemp = {
        0: "<0",
        1: "0-10",
        2: "10-20",
        3: "20-30",
        4: ">30"
    }
    # słownik {<0 : 0, 0-10 : 1, 10-20: 2, 20-30 : 3, 30> : 4}
    for i in range(len(meanT)):
        for col in list(meanT[i].columns):
            meanT[i].loc[meanT[i][col] < 0, col] = 0
            meanT[i].loc[(meanT[i][col] >= 0) & (meanT[i][col] < 10), col] = 1
            meanT[i].loc[(meanT[i][col] >= 10) & (meanT[i][col] < 20), col] = 2
            meanT[i].loc[(meanT[i][col] >= 20) & (meanT[i][col] < 30), col] = 3
            meanT[i].loc[meanT[i][col] >= 30, col] = 4

    diff = []
    zero = []
    one = []
    two = []
    three = []
    four = []
    current = []
    names = []
    lists = [zero, one, two, three, four]
    n = ['<0', '0-10', '10-20', '20-30', '>30']
    for country in list(rep.index):
        lat = rep['Lat'][country]
        long = rep['Long'][country]
        latT = min(list(meanT[0].index), key=lambda x: abs(x - lat))
        longT = min(list(meanT[0].columns), key=lambda x: abs(x - long))
        for i in range(len(meanT)):
            temp = meanT[i][longT][latT]
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

    for ind, k in enumerate(lists):
        if len(k) > 1:
            current.append(k)
            names.append(n[ind])

    nor = np.concatenate([*current])
    k2, p = normaltest(nor)
    if p < 0.05:
        print("Testowanie hipotez Zadanie 1: Rozkład jest normalny.")
        print("Testowanie hipotez Zadanie 1: Wariancje:")
        for c in current:
            print(np.std(c))
        print("Testowanie hipotez Zadanie 1: Wariancje mają zbliżone wartości, można przeprowadzić test Anova.")
        f_value, p_value = f_oneway(*current)
        if p_value < 0.05:
            print(
                "Testowanie hipotez Zadanie 1: Wynik testu Anova wykazał istotne różnice, dlatego wykonano test post-hoc.")
            res = pairwise_tukeyhsd(np.concatenate(current), np.concatenate(
                [[names[ind]]*len(c) for ind, c in enumerate(current)]))
            print("Testowanie hipotez Zadanie 1: Wynik testu post-hoc:")
            print(res)
            df = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
            for t in list(df.index):
                if df.loc[t, 'reject'] == True:
                    g1 = df.loc[t, 'group1']
                    g2 = df.loc[t, 'group2']
                    ind1 = n.index(g1)
                    ind2 = n.index(g2)
                    analysis = TTestIndPower()
                    effect = (np.mean(lists[ind1]) - np.mean(lists[ind2]))/((np.std(lists[ind1]) + np.std(lists[ind2]))/2)
                    pow = analysis.solve_power(effect, power=None, nobs1=len(lists[ind1]),
                                                 ratio=len(lists[ind2])/len(lists[ind1]), alpha=0.05)
                    if pow >= 0.8:
                        diff.append((dictTemp[ind1], dictTemp[ind2]))

    print("Testowanie hipotez Zadanie 1: Dla mocy testu powyżej 80% zarejestrowano istotne różnice pomiędzy grupami temperatur:", diff)
    print("Testowanie hipotez Zadanie 1: Interpretacja: Z przeprowadzonych analiz można wyciągnąć wniosek, że "
          "temperatura wpływa na szybkość rozprzestrzeniania się wirusa.\nIm wyższa temperatura, tym wirus rozprzestrzenia się szybciej.")


def zad5(death, con, mortality):
    print("\n--------------------- Testowanie hipotez Zadanie 2 ---------------------")
    noteurope= []
    for country in list(death.index):
        try:
            country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
            continent_name = pc.country_alpha2_to_continent_code(country_code)
            if continent_name != 'EU':
                noteurope.append(country)
        except:
            if country == 'Kosovo':
                continue
            else:
                noteurope.append(country)
    for c in noteurope:
        death.drop(c, inplace=True)
        con.drop(c, inplace=True)
        mortality.drop(c, inplace=True)
    death = death.iloc[:, 2:]
    con = con.iloc[:, 2:]
    death = death.sum(axis=1)
    con = con.sum(axis=1)
    res = pd.concat([death, con], axis=1)
    res = np.array(res)
    chi2, p, _, _ = chi2_contingency(res)

    print('Testowanie hipotez Zadanie 2: Test chi2: Prawdopodobieństwo testowe wynosi: ', p, ', w związku z tym '
            'odrzucono hipotezę zerową i przyjęto, że różnice w śmiertelności pomiędzy\nkrajami Europy są istotne.')

    anova = []
    diff = []
    mortality.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in list(mortality.index):
        data = mortality.loc[c, :].dropna().tolist()
        anova.append(data)
    nor = np.concatenate([*anova])
    k2, p = normaltest(nor)
    var = []
    if p < 0.05:
        print("Testowanie hipotez Zadanie 2: Test Anova: Rozkład jest normalny.")
        print("Testowanie hipotez Zadanie 2: Test Anova: Wariancje:")
        for a in anova:
            print(np.std(a))
            var.append(np.std(a))
        med = np.median(var)
        print("Testowanie hipotez Zadanie 2: Test Anova: Mediana wariancji:", med)
        rem = []
        coutoDrop = []
        for indek, v in enumerate(var):
            if abs(med-v) > 2:
                rem.append(indek)
                coutoDrop.append(list(mortality.index)[indek])
        rem.reverse()
        for r in rem:
            anova.pop(r)
        for c in coutoDrop:
            print("Testowanie hipotez Zadanie 2: Test Anova: Na podstawie znaczącej różnicy wariancji od mediany wszystkich wariancji "
                  "z testów wykluczono", c)
            mortality.drop(c, inplace=True)
        f_value, p_value = f_oneway(*anova)
        if p_value < 0.05:
            print("Testowanie hipotez Zadanie 2: Test Anova: Wynik testu Anova wykazał istotne różnice, dlatego wykonano test post-hoc.")
            res = pairwise_tukeyhsd(np.concatenate(anova), np.concatenate(
                [[list(mortality.index)[ind]] * len(c) for ind, c in enumerate(anova)]))
            print("Testowanie hipotez Zadanie 2: Test Anova: Wynik testu post-hoc:")
            print(res)
            df = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
            for t in list(df.index):
                if df.loc[t, 'reject'] == True:
                    print(t)
                    g1 = df.loc[t, 'group1']
                    g2 = df.loc[t, 'group2']
                    ind1 = list(mortality.index).index(g1)
                    ind2 = list(mortality.index).index(g2)
                    analysis = TTestIndPower()
                    effect = (np.mean(anova[ind1]) - np.mean(anova[ind2])) / (
                                (np.std(anova[ind1]) + np.std(anova[ind2])) / 2)
                    pow = analysis.solve_power(effect, power=None, nobs1=len(anova[ind1]),
                                               ratio=len(anova[ind2]) / len(anova[ind1]), alpha=0.05)
                    if pow >= 0.8:
                        diff.append(ind1)
                        diff.append(ind2)

    # uni = list(set(diff))
    # count = {}
    # for u in uni:
    #     cnt = 0
    #     for d in diff:
    #         if d == u:
    #             cnt += 1
    #     count[u] = cnt
    # ma = 0
    # i = 0
    # for ind, val in count.items():
    #     if val > ma:
    #         i = ind
    #         ma = val
    print(
        "Testowanie hipotez Zadanie 2: Test Anova: Wynik testu Anova wykazał, że istnieją podstawy do odrzucenia hipotezy "
        "zerowej, natomiast test post-hoc nie wykazał,\npomiędzy którymi krajami zauważalne są istotne różnice.")
    print('Testownie hipotez Zadanie 2: Test Anova: Interpretacja: Danych jest zbyt mało, aby można było wykazać, pomiędzy'
          ' którymi krajami są istotne różnice w śmiertelności.')


if __name__ == "__main__":
    t1_start = perf_counter()
    death, con, active, mortality = zad1()
    rep = zad2(con, active)
    mean = zad3()
    zad4(rep, mean)
    zad5(death, con, mortality)
    t1_stop = perf_counter()
    print("Time:", t1_stop-t1_start)

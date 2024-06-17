import pandas as pd
from tqdm import tqdm
import numpy as np
covid = pd.read_csv('Covid.csv')
#%%
def CountMissingCol(df, col):
    return list(df[col].isnull()).count(True)
#%%
def CountMissingData(df):
    d = dict()
    for col in df.columns:
        d[col] = CountMissingCol(df, col)
    return d
#%%
def DropMissing(df, col):
    misses = df[col].isnull()
    m = list()
    for i in range(len(misses)):
        if misses[i]:
            m.append(i)
    df.drop(m, axis=0, inplace=True)

#%%
def PrevRowFilling(data):
    nulls = data.isnull()
    pRow = 0
    l = list()
    for i in range(len(data)):
        if not nulls[i]:
            l.append(data[i])
            pRow = data[i]
        else:
            l.append(pRow)
    return l
#%%
def NextRowFilling(data):
    nulls = data.isnull()
    nRow = 0
    l = list()
    for i in range(len(data)-1,-1,-1):
        if not nulls[i]:
            l.append(data[i])
            nRow = data[i]
        else:
            l.append(nRow)
    l.reverse()
    return l
#%%
miss = CountMissingData(covid)
#%%
import csv
a_file = open("miss.csv", "w", newline="")

writer = csv.writer(a_file)
for key, value in miss.items():
    writer.writerow([key, value])

a_file.close()
#%%
#Fill in the only country without population data
NCyprusPop = 360000
nulls = covid['population'].isnull()
pop = list()
for i in tqdm(range(len(covid))):
    if nulls[i]:
        if covid.iloc[i]['location'] == 'Northern Cyprus':
            pop.append(NCyprusPop)
        else:
            pop.append(0)
    else:
        pop.append(covid.iloc[i]['population'])
covid['population'] = pop
del i
del pop
del NCyprusPop
del nulls
#%%
nulls = covid['continent'].isnull()
continent = list()
for i in range(len(nulls)):
    if nulls[i]:
        continent.append(covid.iloc[i]['location'])
    else:
        continent.append(covid.iloc[i]['continent'])
covid['continent'] = continent
del i
del continent
del nulls
#%%
InterpolateCols = ['new_cases', 'new_deaths', 'new_tests', 'new_tests_per_thousand', 'hosp_patients', 'icu_patients', 'new_deaths_smoothed', 'new_cases_per_million', 'new_cases_smoothed',
                   'new_cases_smoothed_per_million', 'new_deaths_per_million', 'new_deaths_smoothed', 'new_deaths_smoothed_per_million', 'tests_units',
                   'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'positive_rate', 'tests_per_case',  'hosp_patients_per_million',
                   'hosp_patients', 'icu_patients_per_million', 'icu_patients', 'weekly_hosp_admissions_per_million', 'weekly_hosp_admissions',
                   'excess_mortality_cumulative', 'excess_mortality_cumulative_absolute', 'excess_mortality', 'excess_mortality_cumulative_per_million',
                   'weekly_icu_admissions_per_million', 'weekly_icu_admissions', 'life_expectancy']
NextRow = [ 'total_tests_per_thousand', 'total_cases', 'total_cases_per_million', 'total_deaths', 'new_deaths_per_million', 
               'total_deaths_per_million', 'new_vaccinations', 'new_vaccinations_smoothed', 'new_vaccinations_smoothed_per_million',
               'new_people_vaccinated_smoothed_per_hundred', 'new_people_vaccinated_smoothed', 'total_tests', 'total_tests_per_thousand', 'total_vaccinations',
               'total_vaccinations_per_hundred', 'people_vaccinated', 'people_vaccinated_per_hundred', 'people_fully_vaccinated', 'people_fully_vaccinated_per_hundred',
               'total_boosters', 'total_boosters_per_hundred']
MeanFilling = ['handwashing_facilities', 'population_density', 'diabetes_prevalence', 'gdp_per_capita', 'median_age', 
               'aged_65_older', 'aged_70_older', 'cardiovasc_death_rate', 'human_development_index', 'stringency_index', 'reproduction_rate',
               'hospital_beds_per_thousand', 'female_smokers', 'male_smokers', 'extreme_poverty']
for c in tqdm(InterpolateCols):
    covid[c] = PrevRowFilling(covid[c].interpolate(method='linear'))
for c in tqdm(NextRow):
    covid[c] = NextRowFilling(covid[c])
for c in tqdm(MeanFilling):
    covid[c] = covid[c].fillna(covid[c].mean())
del c

#%%
#1.2
countries = list(set(covid['location']))
new_cases = dict()
new_vaccination = dict()
new_deaths = dict()
population = dict()
for c in countries:
    new_cases[c] = 0
    new_vaccination[c] = 0
    new_deaths[c] = 0
    population[c] = 0
for i in tqdm(range(len(covid))):
    new_cases[covid.iloc[i]['location']] = new_cases[covid.iloc[i]['location']] + covid.iloc[i]['new_cases']
    new_vaccination[covid.iloc[i]['location']] = new_vaccination[covid.iloc[i]['location']] + covid.iloc[i]['new_vaccinations']
    new_deaths[covid.iloc[i]['location']] = new_deaths[covid.iloc[i]['location']] + covid.iloc[i]['new_deaths']
    population[covid.iloc[i]['location']] = int(covid.iloc[i]['population'])
del c
del i
#%%
CountriesDataframe = pd.DataFrame(data=None)
nc = list()
nd = list()
nv = list()
p = list()
WrongLocs = ['Lower middle income', 'North America', 'South America', 'Upper middle income', 'High income', 'Europe', 'European Union',
             'Asia', 'Africa', 'World', 'International', 'Central African Republic', 'Low income', 'Oceania']
CountriesDataframe['Location'] = [c for c in countries if not c in WrongLocs]
for i in range(len(CountriesDataframe)):
    c = CountriesDataframe.iloc[i]['Location']
    nc.append(new_cases[c])
    nv.append(new_vaccination[c])
    nd.append(new_deaths[c])
    p.append(population[c])
CountriesDataframe['new_cases'] = nc
CountriesDataframe['new_deaths'] = nd
CountriesDataframe['new_vaccination'] = nv
CountriesDataframe['new_cases'] = p
del countries
del c
del i
del nc
del nd
del nv
del p
del new_cases
del new_vaccination
del new_deaths
del population
del WrongLocs
#%%
CountriesDataframe.to_csv('CountriesDataframe.csv')
#%%
#1.3
import jdatetime as jdt
shamsi = list()
for i in tqdm(range(len(covid))):
    d = str(covid.iloc[i]['date']).split('/')
    shamsi.append(str(jdt.date.fromgregorian(day=int(d[1]), month=int(d[0]), year=int(d[2]))))
covid['shamsi'] = shamsi
del shamsi
del i
del d
#%%
covid.to_csv('new_covid.csv')
#%%
#1.5
IranRows = list()
for i in range(len(covid)):
    if covid.iloc[i]['location'] == 'Iran':
        IranRows.append(i)
IranData = covid.iloc[IranRows]
del IranRows
del i
#%%
#1.6
month = list()
for i in range(len(IranData)):
    month.append(int(IranData.iloc[i]['shamsi'].split('-')[1]))
IranData['month'] = month
del i
del month
#%%
IranData.to_csv('IranData.csv')
#%%
#1.7
def SumByMonth(df, col):
    v = np.zeros((12,)) 
    for i in range(1,13):
        t = list(df[df['month'] == i][col])
        try:
            v[i-1] = sum(t)
        except:
            return ['' for i in range(12)]
    return v
def MeanByMonth(df, col):
    v = np.zeros((12,)) 
    for i in range(1,13):
        t = list(df[df['month'] == i][col])
        v[i-1] = sum(t)/len(t)
    return v
def ReturnSame(df, col):
    return [df.iloc[0][col] for i in range(12)]
#%%
IranByMonth = pd.DataFrame(data=None)
IranByMonth['month'] = list(range(1,13))
for c in ['location','population','iso_code','continent']:
    IranByMonth[c] = ReturnSame(IranData, c)
for c in NextRow + InterpolateCols:
    IranByMonth[c] = SumByMonth(IranData, c)
for c in MeanFilling:
    IranByMonth[c] = MeanByMonth(IranData, c)
#%%
IranByMonth.to_csv('IranByMonth.csv')
# 123OFAI_Hackthon

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
import sklearn
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
```

### Data Pre-processing


```python
# Reading the Training Data
df = pd.read_csv("test_set_nogt.csv") #, keep_default_na=False) 
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holiday</th>
      <th>temp</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
      <th>weather_description</th>
      <th>date_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>289.58</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>02-10-2012 11:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>290.13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>02-10-2012 12:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>291.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>Clouds</td>
      <td>broken clouds</td>
      <td>02-10-2012 13:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>291.72</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>sky is clear</td>
      <td>02-10-2012 14:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>281.18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>sky is clear</td>
      <td>03-10-2012 02:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.replace('','NaN',regex=True,inplace=True)
```


```python
df.isna().sum()
```




    holiday                9625
    temp                      0
    rain_1h                   0
    snow_1h                   0
    clouds_all                0
    weather_main              0
    weather_description       0
    date_time                 0
    dtype: int64



#### Holiday


```python
# Change holiday column to be a boolean: 1 if holiday else 0
df["holiday_bool"] = np.where(df.holiday=="NaN", 0, 1)
```

#### Temperature and Rain



```python
df['temp'].head()
```




    0    289.58
    1    290.13
    2    291.14
    3    291.72
    4    281.18
    Name: temp, dtype: float64




```python
df['temp'] = df['temp'] - 273.15
```


```python
# drop outliers from temp and rain
df.drop(df[df.temp < -50].index, inplace=True)
df.drop(df[df.rain_1h > 9000].index, inplace=True)

```


```python
df['temp'].isna().sum()
```




    0




```python
plt.figure(figsize= (10,5))
plt.subplot(1,2,1)
sns.boxplot(x='temp', data = df)
plt.subplot(1,2,2)
sns.boxplot(x='rain_1h', data = df)
plt.show()

```


    
![png](Hackathon_Template_Pre_Processing_Model_files/Hackathon_Template_Pre_Processing_Model_14_0.png)
    


#### Date_time


```python
df['date_time'].isna().sum()
```




    0




```python
# convert date_time column to datetime type
#from irdatacleaning import StringToDateTime
df.date_time = pd.to_datetime(df.date_time, dayfirst=True)
#df.date_time = datetime.strptime(df['date_time'],'%d-%m-%Y %H:%M')
#df.date_time = StringToDateTime(df['date_time'])
#df.date_time = df['date_time'].astype('datetime64[ns]')
df['date_time'].isna().sum()
```




    0




```python
# drop datetime if NaN
df.dropna(subset=['date_time'], inplace=True, axis=0)
```


```python
df.date_time.dtype
```




    dtype('<M8[ns]')




```python
df['date_time'].head()
```




    0   2012-10-02 11:00:00
    1   2012-10-02 12:00:00
    2   2012-10-02 13:00:00
    3   2012-10-02 14:00:00
    4   2012-10-03 02:00:00
    Name: date_time, dtype: datetime64[ns]



After transforming the variable into date_time format, we can obtain the years, months, days and
hours from it.


##### Year


```python
# extract year feature
years = df.date_time.dt.year
years.value_counts()
```




    date_time
    2017    2083
    2016    1903
    2013    1710
    2018    1581
    2014     973
    2015     894
    2012     497
    Name: count, dtype: int64




```python
time = pd.DataFrame({
'years' : years,
'traffic_volume' : df.traffic_volume
})
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_11240\2748374660.py in ?()
          1 time = pd.DataFrame({
          2 'years' : years,
    ----> 3 'traffic_volume' : df.traffic_volume
          4 })
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'traffic_volume'



```python
plt.figure(figsize=(8,6))
sns.lineplot(x='years', y='traffic_volume', data= time)
plt.show()
```


    
![png](Hackathon_Template_Pre_Processing_Model_files/Hackathon_Template_Pre_Processing_Model_25_0.png)
    


After examining the traffic volume for each year, we can see a decrease in traffic_volume which
occurs around the end of 2015 - beginning of 2016. This could also be due to a lack of data collection
in that period since during the other years the traffic volume remains rather stable.

##### Month


```python
# extract month feature
months = df.date_time.dt.month
months.value_counts()
```




    date_time
    7     949
    8     878
    5     866
    4     855
    9     839
    12    811
    1     790
    11    778
    6     753
    3     728
    10    711
    2     683
    Name: count, dtype: int64




```python
time = pd.DataFrame({
'years' : years,
'months': months,
'traffic_volume' : df.traffic_volume
})
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_11240\744181453.py in ?()
          1 time = pd.DataFrame({
          2 'years' : years,
          3 'months': months,
    ----> 4 'traffic_volume' : df.traffic_volume
          5 })
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'traffic_volume'



```python
plt.figure(figsize=(8,5))
sns.lineplot(x='months', y='traffic_volume', data= time)
plt.show()
```


    
![png](Hackathon_Template_Pre_Processing_Model_files/Hackathon_Template_Pre_Processing_Model_30_0.png)
    


The traffic volume begins to grow from January until a positive peak in June. A sharp decrease
follows in July. Then there is a new increase during the end of the summer period and it starts to
decrease again during the beginning of the winter months.

##### Day of Month



```python
# extract day of month feature
day_of_months = df.date_time.dt.day
day_of_months.value_counts()
```




    date_time
    6     361
    16    357
    25    348
    4     345
    24    341
    18    338
    14    335
    19    332
    15    330
    30    325
    10    322
    9     320
    17    319
    12    316
    20    315
    21    315
    23    314
    5     309
    13    308
    28    306
    11    306
    22    305
    8     301
    3     295
    26    295
    2     294
    1     291
    7     291
    27    289
    29    267
    31    151
    Name: count, dtype: int64




```python
time = pd.DataFrame({
'years' : years,
'months': months,
'day_of_month':day_of_months,
'traffic_volume' : df.traffic_volume
})
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_11240\4202049444.py in ?()
          1 time = pd.DataFrame({
          2 'years' : years,
          3 'months': months,
          4 'day_of_month':day_of_months,
    ----> 5 'traffic_volume' : df.traffic_volume
          6 })
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'traffic_volume'



```python
plt.figure(figsize=(8,6))
sns.lineplot(x='day_of_month', y='traffic_volume', data= time)
plt.show()
```


    
![png](Hackathon_Template_Pre_Processing_Model_files/Hackathon_Template_Pre_Processing_Model_35_0.png)
    


In this case we notice a rather stable trend in traffic, which remains between the values of 3100
and 3300, with a brief peak towards the end of the month.

##### Day of Week


This time I process the data differently because the goal is to extract the day name.
The process consists of two steps:
- First is to extract the day name literal using pd.Series.dt.day_name() method.
- Afterwards, we need to one-hot encode the results from the first step using pd.get_dummies()
method.


```python
# first: extract the day name literal
days_name = df.date_time.dt.day_name()
# second: one hot encode to 7 columns
days = pd.get_dummies(days_name, dtype=int)
days = days[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]
days

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Monday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>Thursday</th>
      <th>Friday</th>
      <th>Saturday</th>
      <th>Sunday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9636</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9637</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9638</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9639</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9640</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>9641 rows × 7 columns</p>
</div>




```python
time = pd.DataFrame({
'years' : years,
'months': months,
'day_of_month':day_of_months,
'days_name' : days_name,
'traffic_volume' : df.traffic_volume
})

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_11240\1997766983.py in ?()
          2 'years' : years,
          3 'months': months,
          4 'day_of_month':day_of_months,
          5 'days_name' : days_name,
    ----> 6 'traffic_volume' : df.traffic_volume
          7 })
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'traffic_volume'



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='days_name',y='traffic_volume', data = time,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.show()

```


    
![png](Hackathon_Template_Pre_Processing_Model_files/Hackathon_Template_Pre_Processing_Model_41_0.png)
    


The traffic volume begins to grow from the first day of the week, Monday, until Friday. During the
weekend the volume lowers a lot, especially on Sundays.

##### Hour


```python
# extract hour feature
hours = df.date_time.dt.hour
hours.value_counts()
```




    date_time
    8     436
    7     428
    9     426
    23    423
    10    420
    0     419
    21    418
    2     414
    12    411
    11    410
    4     404
    16    400
    1     398
    14    395
    17    394
    6     394
    22    392
    3     391
    18    391
    13    389
    5     386
    19    385
    20    381
    15    336
    Name: count, dtype: int64



This time I will create a grouping based on the hour digits. Six groups representing each daypart:
- Dawn (02.00 — 05.59), 
- Morning (06.00 —09.59), 
- Noon (10.00–13.59), 
- Afternoon (14.00–17.59),
- Evening (18.00–21.59), and 
- Midnight (22.00–01.59 on Day+1). 

To this end, we create an identifying
function that we later use to feed an apply method of a Series. Afterwards, we perform one-hot
encoding on the resulted dayparts.


```python
# daypart function
def day_part(hours):
    if hours in [2,3,4,5]:
        return "dawn"
    elif hours in [6,7,8,9]:
        return "morning"
    elif hours in [10,11,12,13]:
        return "noon"
    elif hours in [14,15,16,17]:
        return "afternoon"
    elif hours in [18,19,20,21]:
        return "evening"
    else: return "midnight"
```


```python
# utilize it along with apply method
day_part = hours.apply(day_part)
```


```python
time = pd.DataFrame({
'years' : years,
'months': months,
'day_of_month':day_of_months,
'days_name' : days_name,
'day_part' :day_part,
'traffic_volume' : df.traffic_volume
})
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_31388\1617521084.py in ?()
          3 'months': months,
          4 'day_of_month':day_of_months,
          5 'days_name' : days_name,
          6 'day_part' :day_part,
    ----> 7 'traffic_volume' : df.traffic_volume
          8 })
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'traffic_volume'



```python
plt.figure(figsize=(8,6))
sns.boxplot(x='day_part', y='traffic_volume', data= time, order=['dawn','morning','noon','afternoon','evening','midnight'])
plt.show()
```


    
![png](Hackathon_Template_Pre_Processing_Model_files/Hackathon_Template_Pre_Processing_Model_49_0.png)
    


The major traffic volume are registerd during morning and afternoon, while very small values are
registered during midnight and dawn.


```python
# one hot encoding
day_part = pd.get_dummies(day_part, dtype=int)
# re-arrange columns for convenience
day_part = day_part[['dawn','morning','noon','afternoon','evening','midnight']]
#display data
day_part
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dawn</th>
      <th>morning</th>
      <th>noon</th>
      <th>afternoon</th>
      <th>evening</th>
      <th>midnight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9636</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9637</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9638</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9639</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9640</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9641 rows × 6 columns</p>
</div>



#### Weather


```python
# dropping weather where holiday is NaN
#df.dropna(subset=['weather_main','holiday'])
# droping thr traffic_volume, holiday and weather main which are having NaN 
#df.dropna(subset=['traffic_volume','holiday','weather_main'],inplace=True,axis=0)
# one-hot encode weather
weathers = pd.get_dummies(df.weather_main, dtype=int)#,dummy_na=True)
#display data
weathers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Clear</th>
      <th>Clouds</th>
      <th>Drizzle</th>
      <th>Fog</th>
      <th>Haze</th>
      <th>Mist</th>
      <th>Rain</th>
      <th>Smoke</th>
      <th>Snow</th>
      <th>Squall</th>
      <th>Thunderstorm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9636</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9637</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9638</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9639</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9640</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9641 rows × 11 columns</p>
</div>




```python
df['temp'].isna().sum()
```




    0




```python
df['rain_1h'].isna().sum()
```




    0




```python
df['snow_1h'].isna().sum()
```




    0




```python
df['clouds_all'].isna().sum() 
```




    0




```python
years.isna().sum()
```




    0




```python
months.isna().sum()
```




    0




```python
day_of_months.isna().sum()
```




    0




```python
days.isna().sum()
```




    Monday       0
    Tuesday      0
    Wednesday    0
    Thursday     0
    Friday       0
    Saturday     0
    Sunday       0
    dtype: int64




```python
day_part.isna().sum()
```




    dawn         0
    morning      0
    noon         0
    afternoon    0
    evening      0
    midnight     0
    dtype: int64




```python
weathers.isna().sum()
```




    Clear           0
    Clouds          0
    Drizzle         0
    Fog             0
    Haze            0
    Mist            0
    Rain            0
    Smoke           0
    Snow            0
    Squall          0
    Thunderstorm    0
    dtype: int64




```python
df.traffic_volume.isna().sum()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_11240\1127212907.py in ?()
    ----> 1 df.traffic_volume.isna().sum()
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'traffic_volume'


# Final Dataset

Finally, I created a new dataset which include all the transformed variables. It will be composed
by 48193 rows and 33 columns.


```python
 #features to keep with just one column of values
features = pd.DataFrame({
'holiday' :df.holiday_bool,
'temp' : df.temp,
'rain_1h' : df.rain_1h,
'snow_1h' :df.snow_1h,
'clouds_all' : df.clouds_all,
'years' : years,
'months':months,
'day_of_month' : day_of_months
})
```


```python
#concat with one-hot encode typed features
features = pd.concat([features, days,day_part, weathers], axis = 1)
```


```python
features.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9641 entries, 0 to 9640
    Data columns (total 32 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   holiday       9641 non-null   int32  
     1   temp          9641 non-null   float64
     2   rain_1h       9641 non-null   float64
     3   snow_1h       9641 non-null   float64
     4   clouds_all    9641 non-null   int64  
     5   years         9641 non-null   int32  
     6   months        9641 non-null   int32  
     7   day_of_month  9641 non-null   int32  
     8   Monday        9641 non-null   int32  
     9   Tuesday       9641 non-null   int32  
     10  Wednesday     9641 non-null   int32  
     11  Thursday      9641 non-null   int32  
     12  Friday        9641 non-null   int32  
     13  Saturday      9641 non-null   int32  
     14  Sunday        9641 non-null   int32  
     15  dawn          9641 non-null   int32  
     16  morning       9641 non-null   int32  
     17  noon          9641 non-null   int32  
     18  afternoon     9641 non-null   int32  
     19  evening       9641 non-null   int32  
     20  midnight      9641 non-null   int32  
     21  Clear         9641 non-null   int32  
     22  Clouds        9641 non-null   int32  
     23  Drizzle       9641 non-null   int32  
     24  Fog           9641 non-null   int32  
     25  Haze          9641 non-null   int32  
     26  Mist          9641 non-null   int32  
     27  Rain          9641 non-null   int32  
     28  Smoke         9641 non-null   int32  
     29  Snow          9641 non-null   int32  
     30  Squall        9641 non-null   int32  
     31  Thunderstorm  9641 non-null   int32  
    dtypes: float64(3), int32(28), int64(1)
    memory usage: 1.3 MB
    


```python
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holiday</th>
      <th>temp</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>years</th>
      <th>months</th>
      <th>day_of_month</th>
      <th>Monday</th>
      <th>Tuesday</th>
      <th>...</th>
      <th>Clouds</th>
      <th>Drizzle</th>
      <th>Fog</th>
      <th>Haze</th>
      <th>Mist</th>
      <th>Rain</th>
      <th>Smoke</th>
      <th>Snow</th>
      <th>Squall</th>
      <th>Thunderstorm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>16.43</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>2012</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>16.98</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>2012</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>17.99</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>2012</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>18.57</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>2012</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>8.03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>2012</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
features.columns
```




    Index(['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'years',
           'months', 'day_of_month', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
           'Friday', 'Saturday', 'Sunday', 'dawn', 'morning', 'noon', 'afternoon',
           'evening', 'midnight', 'Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze',
           'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'],
          dtype='object')




```python
features['temp'].isna().sum()
```




    0



#### temp_c


```python
# Handling missing values in temp_c with the temperature of that month and that year average
def avg():
    dic = {}
    for k in features['years'].unique():
        features3 = features[features['years']==k][['months','temp']]
        dic[k] = {}
        for i in features3['months']:
            sum = 0
            features2 = features3[features3['months'] == i].dropna()['temp']
            if(len(list(features2)) <=0):
                continue
            else:
                for j in features2:
                    sum += j
                dic[k][i] = sum/len(list(features2))
    return(dic)
dic=avg()
```


```python
# Extract the years and months where temp_c is null
null_temp_indices = features[features['temp'].isnull()].index

# Replace null values with values from the dictionary
for idx in null_temp_indices:
    year = features.loc[idx, 'years']
    month = features.loc[idx, 'months']
    if year in dic and month in dic[year]:
        features.at[idx, 'temp'] = dic[year][month]
```

#### rain_1h


```python
features['rain_1h'].isna().sum()
```




    0




```python
# Handling missing values in rain_1h with the rain_1h of that month and that year average
def avg():
    dic = {}
    for k in features['years'].unique():
        features3 = features[features['years']==k][['months','rain_1h']]
        dic[k] = {}
        for i in features3['months']:
            sum = 0
            features2 = features3[features3['months'] == i].dropna()['rain_1h']
            if(len(list(features2)) <=0):
                continue
            else:
                for j in features2:
                    sum += j
                dic[k][i] = sum/len(list(features2))
    return(dic)
dic=avg()
```


```python
# Extract the years and months where rain_1h is null
null_temp_indices = features[features['rain_1h'].isnull()].index

# Replace null values with values from the dictionary
for idx in null_temp_indices:
    year = features.loc[idx, 'years']
    month = features.loc[idx, 'months']
    if year in dic and month in dic[year]:
        features.at[idx, 'rain_1h'] = dic[year][month]
```

#### Snow_1h


```python
# Handling missing values in rain_1h with the rain_1h of that month and that year average
def avg():
    dic = {}
    for k in features['years'].unique():
        features3 = features[features['years']==k][['months','snow_1h']]
        dic[k] = {}
        for i in features3['months']:
            sum = 0
            features2 = features3[features3['months'] == i].dropna()['snow_1h']
            if(len(list(features2)) <=0):
                continue
            else:
                for j in features2:
                    sum += j
                dic[k][i] = sum/len(list(features2))
    return(dic)
dic=avg()
```


```python
# Extract the years and months where rain_1h is null
null_temp_indices = features[features['snow_1h'].isnull()].index

# Replace null values with values from the dictionary
for idx in null_temp_indices:
    year = features.loc[idx, 'years']
    month = features.loc[idx, 'months']
    if year in dic and month in dic[year]:
        features.at[idx, 'snow_1h'] = dic[year][month]
```

#### clouds_all


```python
# Handling missing values in rain_1h with the rain_1h of that month and that year average
def avg():
    dic = {}
    for k in features['years'].unique():
        features3 = features[features['years']==k][['months','clouds_all']]
        dic[k] = {}
        for i in features3['months']:
            sum = 0
            features2 = features3[features3['months'] == i].dropna()['clouds_all']
            if(len(list(features2)) <=0):
                continue
            else:
                for j in features2:
                    sum += j
                dic[k][i] = sum/len(list(features2))
    return(dic)
dic=avg()
```


```python
# Extract the years and months where rain_1h is null
null_temp_indices = features[features['clouds_all'].isnull()].index

# Replace null values with values from the dictionary
for idx in null_temp_indices:
    year = features.loc[idx, 'years']
    month = features.loc[idx, 'months']
    if year in dic and month in dic[year]:
        features.at[idx, 'clouds_all'] = dic[year][month]
```

#### traffic_volume


```python
features['traffic_volume'].isna().sum()
```




    0




```python
features['traffic_volume'].fillna(features['traffic_volume'].mean(), inplace=True)
```



# Correlation Matrix


With all the variables in a numerical type, I can perform the Correlation Matrix


```python
#Correlation Matrix, standard method 'Pearson'
f = plt.figure(figsize=(10, 8))
plt.matshow(features.corr(), fignum=f.number)
plt.xticks(range(features.shape[1]), features.columns, fontsize=14, rotation=90)
plt.yticks(range(features.shape[1]), features.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
```


    
![png](Hackathon_Template_Pre_Processing_Model_files/Hackathon_Template_Pre_Processing_Model_92_0.png)
    


There are no strong correlation among all the variables, except the one Clear-Clouds_all and those
of the days of the week.
Even more we note the lack of strong correlations between most of the variables and our target
variable. The exceptions in this case are to be found on the days of the week (especially those of
the weekend) and the part of the day

# Splitting the dataset


```python
from sklearn.model_selection import train_test_split # split the dataset into training and test set
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error #to evaluate r2 score, mse, mae
```


```python
features.columns
```




    Index(['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'years',
           'months', 'day_of_month', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
           'Friday', 'Saturday', 'Sunday', 'dawn', 'morning', 'noon', 'afternoon',
           'evening', 'midnight', 'Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze',
           'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'],
          dtype='object')




```python
X = features.iloc[:,30]
y = features.iloc[:,30]
```

I decided to split the dataset in training set for the 80% and test set for the last 20%.
Notice below I do not shuffle our data, this is due to the time-series nature of the data.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle =False, random_state = 1231)
```

# Normalization

Normalization should be done after splitting the data between training and test set, using only the
data from the training set. This is because the testing data points represent real-world data, so it’s
not supposed to be accessible at the training stage. Using any information coming from the test
set before or during training is a potential bias in the evaluation of the performance.
Therefore, we should perform feature scaling over the training data and then perform normalization
on testing instances but this time using the mean and standard deviation of training explanatory
variables.


```python
from sklearn.preprocessing import MinMaxScaler
# create a scaler object
scaler = MinMaxScaler()
# fit and transform the data
X_train_norm= pd.DataFrame(scaler.fit_transform(X_test.values.reshape(-1,1)) )
```


```python
X_test_norm = pd.DataFrame(scaler.transform(X_test.values.reshape(-1,1)))
```


```python
X_train_norm.isna().sum()
```




    0    0
    dtype: int64




```python
X_test_norm.isna().sum()
```




    0    0
    dtype: int64




```python
X_test_norm.columns
```




    RangeIndex(start=0, stop=1, step=1)




```python
y_test.reshape(-1,1)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_18576\3736495560.py in ?()
    ----> 1 y_test.reshape(-1,1)
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'Series' object has no attribute 'reshape'


# Modelling

### Multiple Linear Regression

Multiple Linear Regression fits a linear model with coefficients to minimize the residual sum of
squares between the observed targets in the dataset, and the targets predicted by the linear approximation.


```python
Model= ['Linear Regression', 'Linear SVR', 'SVR','Decision Tree','Random Forest Regression', 'Gradient Boosting Regression', 'K-Nearest Neighbors']
R_squared =list()
RMSE = list()
MAE = list()

```


```python
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
```


```python
LR = LinearRegression()
LR.fit(X_train_norm,y_train)
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
y_test
```




    array([[0, 0, 0, ..., 0, 0, 0]])




```python
X_train_norm.columns
```




    RangeIndex(start=0, stop=1, step=1)




```python
test = pd.read_csv("")
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[57], line 1
    ----> 1 test = pd.read_csv("")
    

    File ~\AppData\Roaming\Python\Python311\site-packages\pandas\io\parsers\readers.py:948, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
        935 kwds_defaults = _refine_defaults_read(
        936     dialect,
        937     delimiter,
       (...)
        944     dtype_backend=dtype_backend,
        945 )
        946 kwds.update(kwds_defaults)
    --> 948 return _read(filepath_or_buffer, kwds)
    

    File ~\AppData\Roaming\Python\Python311\site-packages\pandas\io\parsers\readers.py:611, in _read(filepath_or_buffer, kwds)
        608 _validate_names(kwds.get("names", None))
        610 # Create the parser.
    --> 611 parser = TextFileReader(filepath_or_buffer, **kwds)
        613 if chunksize or iterator:
        614     return parser
    

    File ~\AppData\Roaming\Python\Python311\site-packages\pandas\io\parsers\readers.py:1448, in TextFileReader.__init__(self, f, engine, **kwds)
       1445     self.options["has_index_names"] = kwds["has_index_names"]
       1447 self.handles: IOHandles | None = None
    -> 1448 self._engine = self._make_engine(f, self.engine)
    

    File ~\AppData\Roaming\Python\Python311\site-packages\pandas\io\parsers\readers.py:1705, in TextFileReader._make_engine(self, f, engine)
       1703     if "b" not in mode:
       1704         mode += "b"
    -> 1705 self.handles = get_handle(
       1706     f,
       1707     mode,
       1708     encoding=self.options.get("encoding", None),
       1709     compression=self.options.get("compression", None),
       1710     memory_map=self.options.get("memory_map", False),
       1711     is_text=is_text,
       1712     errors=self.options.get("encoding_errors", "strict"),
       1713     storage_options=self.options.get("storage_options", None),
       1714 )
       1715 assert self.handles is not None
       1716 f = self.handles.handle
    

    File ~\AppData\Roaming\Python\Python311\site-packages\pandas\io\common.py:863, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        858 elif isinstance(handle, str):
        859     # Check whether the filename is to be opened in binary mode.
        860     # Binary mode does not support 'encoding' and 'newline'.
        861     if ioargs.encoding and "b" not in ioargs.mode:
        862         # Encoding
    --> 863         handle = open(
        864             handle,
        865             ioargs.mode,
        866             encoding=ioargs.encoding,
        867             errors=errors,
        868             newline="",
        869         )
        870     else:
        871         # Binary mode
        872         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 2] No such file or directory: ''



```python
LR = LinearRegression()
LR.fit(X_test.values.reshape(-1,1),y_test)
y_test = np.array(y_test.values.tolist())

```


```python
y_test_reshaped=y_test.reshape(1,-1)
```


```python
predictions = LR.predict(y_test_reshaped)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[74], line 1
    ----> 1 predictions = LR.predict(y_test_reshaped)
    

    File ~\AppData\Roaming\Python\Python311\site-packages\sklearn\linear_model\_base.py:386, in LinearModel.predict(self, X)
        372 def predict(self, X):
        373     """
        374     Predict using the linear model.
        375 
       (...)
        384         Returns predicted values.
        385     """
    --> 386     return self._decision_function(X)
    

    File ~\AppData\Roaming\Python\Python311\site-packages\sklearn\linear_model\_base.py:369, in LinearModel._decision_function(self, X)
        366 def _decision_function(self, X):
        367     check_is_fitted(self)
    --> 369     X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        370     return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    

    File ~\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py:626, in BaseEstimator._validate_data(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)
        623     out = X, y
        625 if not no_val_X and check_params.get("ensure_2d", True):
    --> 626     self._check_n_features(X, reset=reset)
        628 return out
    

    File ~\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py:415, in BaseEstimator._check_n_features(self, X, reset)
        412     return
        414 if n_features != self.n_features_in_:
    --> 415     raise ValueError(
        416         f"X has {n_features} features, but {self.__class__.__name__} "
        417         f"is expecting {self.n_features_in_} features as input."
        418     )
    

    ValueError: X has 1929 features, but LinearRegression is expecting 1 features as input.



```python
print('R square score on train set and test set are :',LR.score(X_train_norm,y_train),LR.score(X_test_norm,y_test))
print('Root mean squared error :',np.sqrt(mean_squared_error(y_test,LR.predict(X_test_norm))))
print('Mean absolute error :',mean_absolute_error(y_test,LR.predict(X_test_norm)))
```

    R square score on train set and test set are : 0.6950081730314502 0.6820586101590682
    Root mean squared error : 1052.8818048803037
    Mean absolute error : 824.4398419637799
    


```python
R_squared.append(LR.score(X_test_norm, y_test))
RMSE.append(np.sqrt(mean_squared_error(y_test,LR.predict(X_test_norm))))
MAE.append(mean_absolute_error(y_test,LR.predict(X_test_norm)))
```


```python
#predictions = LR.predict(X_test,y_test)
```

### Support Vector Machines

Support vector machines (SVMs) are a set of supervised learning methods used for classification,
regression and outliers detection.
The method to solve regression problems is called Support Vector Regression, it depends only on a
subset of the training data, because the cost function ignores samples whose prediction is close to
their target.


```python
from sklearn import svm
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
```

#### Linear Support Vector Regression

LinearSVR provides a faster implementation than SVR but only considers the linear kernel, but
it has more flexibility in the choice of penalties and loss functions and should scale better to large
numbers of samples.



```python
LinearSVR = LinearSVR(random_state= 1231)
#LinearSVR.fit(X_train_norm,y_train)
LinearSVR.fit(X_test,y_test)
LinearSVR.predict(X_test,y_test)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 3
          1 LinearSVR = LinearSVR(random_state= 1231)
          2 #LinearSVR.fit(X_train_norm,y_train)
    ----> 3 LinearSVR.fit(X_test,y_test)
          4 LinearSVR.predict(X_test,y_test)
    

    NameError: name 'X_test' is not defined



```python
print('R square score on train set and test set are :',LinearSVR.score(X_train_norm,y_train),LinearSVR.score(X_test_norm,y_test))
print('Root mean squared error :',np.sqrt(mean_squared_error(y_test,LinearSVR.predict(X_test_norm))))
print('Mean absolute error :',mean_absolute_error(y_test,LinearSVR.predict(X_test_norm)))

```

    R square score on train set and test set are : 0.6614991304659714 0.6512857884845551
    Root mean squared error : 1102.6581960161204
    Mean absolute error : 882.0069668261871
    


```python

```

Tuning the Hyper-parameters During the building of our models, it is possible and recommended to search the hyper-parameter space for the best cross validation score. Any parameter
provided when constructing an estimator may be optimized in this manner.
The approach I used to parameter search is provided by GridSearchCV, exhaustively generates
candidates from a grid of parameter values specified with the parameter_grid parameter.
In this case, I will evaluate models using the negative mean absolute error
(neg_mean_absolute_error). It is negative because the GridsearchCV requires the score to
be maximized, so the MAE is made negative, meaning scores scale from -infinity to 0 (best).


```python
from sklearn.model_selection import GridSearchCV
```


```python
parameter_grid = {'C': range(1,100)}
GS=GridSearchCV(LinearSVR,parameter_grid,cv=3, scoring='neg_mean_squared_error')
GS.fit(X_train_norm,y_train)
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3, estimator=LinearSVR(random_state=1231),
             param_grid={&#x27;C&#x27;: range(1, 100)}, scoring=&#x27;neg_mean_squared_error&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3, estimator=LinearSVR(random_state=1231),
             param_grid={&#x27;C&#x27;: range(1, 100)}, scoring=&#x27;neg_mean_squared_error&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LinearSVR</label><div class="sk-toggleable__content"><pre>LinearSVR(random_state=1231)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">LinearSVR</label><div class="sk-toggleable__content"><pre>LinearSVR(random_state=1231)</pre></div></div></div></div></div></div></div></div></div></div>




```python
GS.best_params_
```




    {'C': 4}




```python
from sklearn.svm import LinearSVR
```


```python
HLinearSVR = LinearSVR(C=3, random_state=1213)
HLinearSVR.fit(X_train_norm,y_train)
```




<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearSVR(C=3, random_state=1213)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">LinearSVR</label><div class="sk-toggleable__content"><pre>LinearSVR(C=3, random_state=1213)</pre></div></div></div></div></div>




```python
print('R square score on train set and test set are :',HLinearSVR.score(X_train_norm,y_train),HLinearSVR.score(X_test_norm,y_test))
print('Root mean squared error :',np.sqrt(mean_squared_error(y_test,HLinearSVR.predict(X_test_norm))))
print('Mean absolute error :',mean_absolute_error(y_test,HLinearSVR.predict(X_test_norm)))

```

    R square score on train set and test set are : 0.6817211111062997 0.6642508504577063
    Root mean squared error : 1081.9658342114756
    Mean absolute error : 816.6404737727481
    


```python
R_squared.append(HLinearSVR.score(X_test_norm, y_test))
RMSE.append(np.sqrt(mean_squared_error(y_test,HLinearSVR.predict(X_test_norm))))
MAE.append(mean_absolute_error(y_test,HLinearSVR.predict(X_test_norm)))
```

#### Support Vector Regressor


```python
SVR = SVR()
SVR.fit(X_train_norm,y_train)
```




<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVR()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">SVR</label><div class="sk-toggleable__content"><pre>SVR()</pre></div></div></div></div></div>




```python
print('R square score on train set and test set are :',SVR.score(X_train_norm,y_train),SVR.score(X_test_norm,y_test))
print('Root mean squared error :',np.sqrt(mean_squared_error(y_test,SVR.predict(X_test_norm))))
print('Mean absolute error :',mean_absolute_error(y_test,SVR.predict(X_test_norm)))

```


```python
test = pd.read_csv("test_set_nogt.csv")
test.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holiday</th>
      <th>temp</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
      <th>weather_description</th>
      <th>date_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>289.58</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>02-10-2012 11:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>290.13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>02-10-2012 12:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>291.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>Clouds</td>
      <td>broken clouds</td>
      <td>02-10-2012 13:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>291.72</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>sky is clear</td>
      <td>02-10-2012 14:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>281.18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>sky is clear</td>
      <td>03-10-2012 02:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.columns
```




    Index(['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main',
           'weather_description', 'date_time'],
          dtype='object')




```python
test.rename(columns={'temp':'temp_c'}, inplace=True)
```


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holiday</th>
      <th>temp_c</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
      <th>weather_description</th>
      <th>date_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>289.58</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>02-10-2012 11:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>290.13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>02-10-2012 12:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>291.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>Clouds</td>
      <td>broken clouds</td>
      <td>02-10-2012 13:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>291.72</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>sky is clear</td>
      <td>02-10-2012 14:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>281.18</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>Clear</td>
      <td>sky is clear</td>
      <td>03-10-2012 02:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
test[features]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_11240\1131288631.py in ?()
    ----> 1 test[features]
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\frame.py in ?(self, key)
       3879             return self._getitem_slice(key)
       3880 
       3881         # Do we have a (boolean) DataFrame?
       3882         if isinstance(key, DataFrame):
    -> 3883             return self.where(key)
       3884 
       3885         # Do we have a (boolean) 1d indexer?
       3886         if com.is_bool_indexer(key):
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, cond, other, inplace, axis, level)
      10599                         ChainedAssignmentError,
      10600                         stacklevel=2,
      10601                     )
      10602         other = common.apply_if_callable(other, self)
    > 10603         return self._where(cond, other, inplace, axis, level)
    

    ~\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py in ?(self, cond, other, inplace, axis, level)
      10305                     raise ValueError(msg.format(dtype=cond.dtype))
      10306             else:
      10307                 for _dt in cond.dtypes:
      10308                     if not is_bool_dtype(_dt):
    > 10309                         raise ValueError(msg.format(dtype=_dt))
      10310                 if cond._mgr.any_extension_types:
      10311                     # GH51574: avoid object ndarray conversion later on
      10312                     cond = cond._constructor(
    

    ValueError: Boolean array expected for the condition, not object



```python
predictions= LR.predict(test)
predictions
```

    

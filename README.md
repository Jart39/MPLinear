# MPLinear
paper code：MPLinear: Multiscale Patch Linear Model for Long-Term Time Series Forecasting

## Usage
1. install
```
pip install -r requirements.txt
```
2. run demo
```
bash ./scripts/long_term_forecast/Weather_script/MPLinear.sh
```
## Datasets

ETT dataset is transformer data from two different counties in the same province in China. It consists of hourly-level datasets and 15-minute-level datasets. Each of them contains seven oil and load features of electricity transformers from July 2016 to July 2018. It was acquired at https:// github.com/zhouhaoyi/ETDataset.

Weather dataset contains 21 variables such as temperature, pressure and humidity in Germany in 2020. It was acquired at https://www.ncei.noaa.gov/ data/local-climatological-data/.

Electricity dataset contains the hourly electricity consumption of 321 customers from 2012 to 2014. It was acquired at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.

Traffic dataset describes hourly road occupancy measured by different sensors on San Francisco Bay Area highways from 2015 to 2016. It was acquired at http://pems.dot.ca.gov.

Exchange dataset collects daily exchange rates for 8 countries from 1990 to 2016. It was acquired at https:// github.com/zhouhaoyi/ETDataset.

ILI dataset describes the ratio of patients seen with influenzalike illness and the number of patients, including weekly recorded influenzalike illness (ILI) patient data from the Centers for Disease Control and Prevention of the United States between 2002 and 2021. It was acquired at https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html.

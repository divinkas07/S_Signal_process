import pandas as pd
import requests

##scraping data from worldometers.info

source = { 
    'Country': 'https://www.worldometers.info/world-population/population-by-country/',
    'Year': 'https://www.worldometers.info/world-population/population-by-country/',
    'Population': 'https://www.worldometers.info/world-population/population-by-country/',
    'GDP': 'https://www.worldometers.info/gdp/gdp-by-country/',
    'Area': 'https://www.worldometers.info/world-population/population-by-country/'
}

data = pd.DataFrame(columns=['Country', 'Year', 'Population', 'GDP', 'Area'])

print(data)
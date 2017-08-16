
# coding: utf-8



import pandas as pd
import numpy as np
from fbprophet import Prophet

import multiprocessing as mp



print("Loading wikipedia dataframe...")
from wikipedia_kaggle import load_wikipedia_dataframe
wiki_df = load_wikipedia_dataframe()


print("Loading pagekeys dataframe...")
#Dataframe that stores the keys for compact csv files
keys_df = pd.read_csv('../key_1.csv/key_1.csv')







train_date_stamps = list(wiki_df.axes[1][1:-4])

future_df = pd.read_csv('forecast_csv.csv')



def forecast_row(row_series):
    pagename = row_series['Page']
    row_values = row_series.drop(['Page', 'lang', 'access_type', 'agent', 'pagename']).tolist()
    
    train_df = pd.DataFrame(data={'ds':train_date_stamps, 'y':row_values})
    
    m = Prophet(yearly_seasonality=False, uncertainty_samples = 0)
    
    m.fit(train_df) #Train classifier
    forecast = m.predict(future_df)
    

    
    forecast['yhat'] = forecast['yhat'].clip(lower=0, axis=0).round(2) #Ensure negative values go 0, round to save space
    forecast['ds'] = pd.Series(data=[pagename + "_"]*row_series.size) + forecast['ds'].map(lambda a: str(a)[:-9])
    
    return forecast[['ds','yhat']]



def forecast_and_save(fc_index):
    fc = forecast_row(wiki_df.iloc[fc_index])
    #Get index of the first item id
    
    first_index = keys_df.loc[keys_df['Page'] == fc['ds'][0]].index[0]
    
    def get_pagekey(k):
        return keys_dict[k]
    
    #Create keys dict
    keys_dict = dict(keys_df[first_index:first_index + 500].values)
    
    fc['ds'] = fc['ds'].map(get_pagekey)
    fc.to_csv("forecasts/fc_{}.csv".format(fc_index), index=False)
    
    







#from ThreadPool import ThreadPool



#pool = mp.Pool(4)

mp_pool = mp.Pool(4)



def forecast_and_save_range(start, stop):
    for fc_index in range(start, stop):
        print("Working on {}...".format(fc_index))
        try:
            forecast_and_save(fc_index)
        except:
            print("ERROR ON {}".format(fc_index))



            
            
for s in range(50000, 51000, 200):
    start = s
    stop = s + 500
    mp_pool.apply_async(forecast_and_save_range, [start, stop])
    #pool.add_task(forecast_and_save_range, start, stop)

#mp_pool.close()
#mp_pool.join()
#pool.wait_completion()



print("DONE!")


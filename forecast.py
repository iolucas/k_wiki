import pandas as pd
import numpy as np
from fbprophet import Prophet
import multiprocessing as mp
import time
from wikipedia_kaggle import load_wikipedia_dataframe



def forecast_row(row_series, train_date_stamps, future_df):
    pagename = row_series['Page']
    row_values = row_series.drop(['Page', 'lang', 'access_type', 'agent', 'pagename']).tolist()
    train_df = pd.DataFrame(data={'ds':train_date_stamps, 'y':row_values})
    
    m = Prophet(yearly_seasonality=False, uncertainty_samples = 0)
    
    m.fit(train_df) #Train classifier
    forecast = m.predict(future_df)
    
    forecast['yhat'] = forecast['yhat'].clip(lower=0, axis=0).round(2) #Ensure negative values go 0, round to save space
    forecast['ds'] = pd.Series(data=[pagename + "_"]*row_series.size) + forecast['ds'].map(lambda a: str(a)[:-9])
    
    return forecast[['ds','yhat']]


def forecast_and_save(index_info):

    fc_index = index_info[0]
    wiki_df = index_info[1]
    keys_df = index_info[2] 
    train_date_stamps = index_info[3]
    future_df = index_info[4]
    


    print("Working on {}".format(fc_index))
    fc = forecast_row(wiki_df.iloc[fc_index], train_date_stamps, future_df)
    #Get index of the first item id
    
    first_index = keys_df.loc[keys_df['Page'] == fc['ds'][0]].index[0]
    
    def get_pagekey(k):
        return keys_dict[k]
    
    #Create keys dict
    keys_dict = dict(keys_df[first_index:first_index + 500].values)
    
    fc['ds'] = fc['ds'].map(get_pagekey)
    fc.to_csv("forecasts/fc_{}.csv".format(fc_index), index=False)
    
if __name__ == "__main__":   

    print("Loading wikipedia dataframe...")
    wiki_df = load_wikipedia_dataframe()

    #Dataframe that stores the keys for compact csv files
    print("Loading page keys dataframe...")
    keys_df = pd.read_csv('../key_1.csv/key_1.csv')

    train_date_stamps = list(wiki_df.axes[1][1:-4])

    future_df = pd.read_csv('forecast_csv.csv')

    indexes_to_go = [(i, wiki_df, keys_df, train_date_stamps, future_df) for i in range(80000,90000)]

    #1 thread took 335 seconds

    #with mp.Pool(8) as pool:
    start_time = time.time()
    for ind_inf in indexes_to_go:
        try:
            forecast_and_save(ind_inf)
        except:
            pass
    #map(forecast_and_save, indexes_to_go)
    end_time = time.time() - start_time
    print(end_time)


    print("DONE!")


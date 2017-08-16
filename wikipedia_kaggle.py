import pandas as pd

def load_wikipedia_dataframe():
    #Load csv
    data_path = '../train_1.csv/train_1.csv'
    wiki_df = pd.read_csv(data_path)
    
    
    #Get page encoded info
    pagenames = []
    pagelangs = []
    access_types = []
    agents = []

    for page in wiki_df['Page']:
        #Parse page data from page string
        pagename, pagelang, access_type, agent = _get_page_info(page)

        pagenames.append(pagename)
        pagelangs.append(pagelang)
        access_types.append(access_type)
        agents.append(agent)
    
    wiki_df['pagename'] = pd.Series(pagenames, index=wiki_df.index)
    wiki_df['lang'] = pd.Series(pagelangs, index=wiki_df.index)
    wiki_df['access_type'] = pd.Series(access_types, index=wiki_df.index)
    wiki_df['agent'] = pd.Series(agents, index=wiki_df.index)
    
    
    #Generate dummy variables
    # dummy_fields = ['lang', 'access_type', 'agent']
    # for each in dummy_fields:
    #     dummies = pd.get_dummies(wiki_df[each], prefix=each, drop_first=False)
    #     wiki_df = pd.concat([wiki_df, dummies], axis=1)
    
    
    return wiki_df
    
    
    
def _get_page_info(page):
    underscore_split = page.split("_")
    
    agent = underscore_split[-1]
    access_type = underscore_split[-2]
    pagelang = underscore_split[-3].split(".")[0]
    pagename = "_".join(underscore_split[:-3])
    
    return pagename, pagelang, access_type, agent
    
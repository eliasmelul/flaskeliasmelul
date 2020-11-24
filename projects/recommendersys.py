import pandas as pd
from scipy.spatial.distance import cosine

norm_df = pd.read_csv('https://s3.us-east-2.amazonaws.com/www.findingmyschittscreek.com/Data/normalized_df_sub.csv', index_col=0)

def from_city_cosSim(data, name):
    Xs = data[data.City == name].drop('City',1)
    Col_A = data[data.City != name].City
    Ys = data[data.City != name].drop('City', 1)
    
    cos_list = []
    for i, row in Ys.iterrows():
        cos_list.append((1-cosine(Xs, row)))
        
    cos_df = pd.DataFrame({"City": list(Col_A), "Score": cos_list})
    cos_df = cos_df.append({"City":'Los Angeles, CA', "Score":1}, ignore_index=True).reset_index(drop=True).sort_values('Score', ascending=True)
    
    return cos_df

def input_cities(cities, numShow=10):
    
    cosSim = pd.DataFrame()
    for city in cities:
        simSim = from_city_cosSim(data=norm_df, name=city)
        if cosSim.empty:
            cosSim = simSim
        else:
            cosSim = pd.concat([cosSim, simSim])
    
    scores = cosSim.groupby('City').mean()
    scores = scores.sort_values('Score', ascending=False)
    topX = scores.iloc[:numShow]
    topX_list = [topX.index[i]+" ("+str( round(topX["Score"][i],2) )+")" for i in range(len(topX))]
    
    return topX_list
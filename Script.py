# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:39:03 2017

@author: kgairola
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")


def main():
    
    df_invt_current = pd.read_excel("Inventory_Current_Onsite.XLSX")
    df_invt_hist = pd.read_excel("Inventory_Historical.xlsx")
    df_kw = pd.read_excel("KW_Attributes.xlsx")
    df_kw_hist = pd.read_excel("KW_Performance_L120D.XLSX")
    df_ARS = pd.read_excel("Make_Model_ASR.XLSX")
    
    df_kw_hist.head()
    df_kw['Year'] = "none"
    df_kw['Car_model'] = "none"
    df_kw['Car_make'] = "none"
    df_kw["Keyword_new"] = "none"
    
    #for i in tqdm.tqdm(range(len(df_kw))):
    #    df_kw['Car_model'][i], df_kw['Car_make'][i] = remove_keywords2(df_kw.iloc[i]["Keyword"])
    
    # Cleaning and extracting data from Keyword column
    df_kw["Keyword"] =df_kw['Keyword'].str.title()
    df_kw["Keyword"] =df_kw['Keyword'].str.replace('+', '')
    df_kw["Keyword"] =df_kw['Keyword'].str.replace("Gmc", "GMC")
    df_kw["Keyword"] =df_kw['Keyword'].str.replace("Crv", "CRV")
    df_kw["Keyword"] =df_kw['Keyword'].str.replace("Srx","SRX")
    df_kw["Keyword"] =df_kw['Keyword'].str.replace("Leaf", "LEAF")
    
    df_kw['Year'] = df_kw.apply(lambda x: re.findall("(\d+)", x["Keyword"])[0], axis= 1)
    from nltk.stem.wordnet import WordNetLemmatizer
    
    Model = ['Camry', 'Soul', 'Civic', 'Corolla', 'Elantra', 'Fusion', 'Terrain','Prius', 'Accord', 'Sonata', 'Optima', 
         'Cruze', 'Altima', 'Verano','Volt', 'CRV', 'Sorento', 'Equinox', 'Camaro', 'SRX', 'Forte', 'LEAF', 'Acadia']
    lem = WordNetLemmatizer() 
    fcn = lambda x: " ".join(set([lem.lemmatize(w, 'v') for w in x]).intersection(set(Model)))
    df_kw["Car_model"] = df_kw['Keyword'].str.split().apply(fcn)
    
    Make = ['Toyota', 'Kia', 'Honda', 'Hyundai', 'Ford', 'GMC', 'Chevrolet', 'Nissan', 'Buick', 'Cadillac']
    fcn = lambda x: " ".join(set([lem.lemmatize(w, 'v') for w in x]).intersection(set(Make)))
    df_kw["Car_make"] = df_kw['Keyword'].str.split().apply(fcn)
    
    df_kw['Keyword_new'] = df_kw.apply(lambda x:'%s%s%s' % (x['Car_make'],x['Car_model'],x['Year']),axis=1)
    
    # Joining df_kw and df_kw_hist ( historical data )
    T1 = pd.merge(df_kw, df_kw_hist, on=df_kw['KW ID'], how='left')
    
    # Joining T1 with df_ARS
    T1 = T1.merge(df_ARS, left_on='Car_model', right_on='Model', how='left')
    
    # Extracting "Market" from "Campaign column"
    T1["Market"] = T1.apply(lambda s: s["Campaign"].replace("SRCH-I-","").replace("-TOTL",""), axis =1)
    
    # Removing all the redundant columns
    T1 = T1.drop(["Campaign", "KW ID_y", "Impressions", "Make" , "Model", "Cost", "Keyword"], axis = 1)
    T1['CVR'] = T1['Conversions']/T1['Clicks']
    T1["New_KW_Bid"] = "None"
    T1["Mk/Mo/Yr OR Mk/Mo CVR"] = 0
    T1 = T1.rename(columns={"KW ID_x": "KW ID"})
    T1["Ad group"] = T1.apply(lambda x : x["Ad group"].strip(), axis = 1)
    
    #Step 1 a
    
    T1["New_KW_Bid"] = T1.apply(lambda x: x["CVR"] * x["ASR"] if x["Conversions"]>=10 else 0, axis = 1)
    #T1[["Conversions","Ad group", "CVR", "ASR", "New_KW_Bid", "AdGrpConversion"]][T1["Conversions"] < 11].head()
    
    #Step - 1 b
    T2 = T1.groupby(['Ad group'])[["Ad group", "Conversions", "Clicks"]].sum()
    T2 = T2.add_suffix('_ad_group').reset_index()
    T2["Ad group"] = T2.apply(lambda x : x["Ad group"].strip(), axis = 1)
    T2["Group_CVR"] = T2.apply(lambda x : x["Conversions_ad_group"]/x["Clicks_ad_group"], axis = 1)
    
    T1 = pd.merge(T1, T2, left_on=T1['Ad group'], right_on = T2["Ad group"], how='left')
    T1 = T1.rename(columns={"Ad group_x": "Ad group"})
    T1 = T1.drop(["Ad group_y"], axis = 1)
    T1.reset_index()
    T1["New_KW_Bid"] = T1.apply(lambda x: x["Group_CVR"] * x["ASR"] if x["Conversions"]<10 and x["Conversions_ad_group"]>=10 else x["New_KW_Bid"], axis = 1)
    T1[["Conversions","Ad group", "CVR", "ASR", "New_KW_Bid"]].head()
    
    # Step 1 - c
    T2 = T1.groupby(['Keyword_new'])[["Keyword_new", "Conversions", "Clicks"]].sum()
    T2 = T2.add_suffix('_Mk_Mo_Yr').reset_index()
    T2["Keyword_new"] = T2.apply(lambda x : x["Keyword_new"].strip(), axis = 1)
    T2["Group_Mk_Mo_Yr_CVR"] = T2.apply(lambda x : x["Conversions_Mk_Mo_Yr"]/x["Clicks_Mk_Mo_Yr"], axis = 1)
    
    T1 = pd.merge(T1, T2, left_on=T1['Keyword_new'], right_on = T2["Keyword_new"], how='left')
    T1 = T1.rename(columns={"Keyword_new_x": "Keyword_new"})
    T1 = T1.drop(["Keyword_new_y"], axis = 1)
    T1.reset_index()
    T1["New_KW_Bid"] = T1.apply(lambda x: x["Group_Mk_Mo_Yr_CVR"] * x["ASR"] if x["Conversions_ad_group"]<11 and x["Conversions_Mk_Mo_Yr"]>=10 else x["New_KW_Bid"], axis = 1)
    T1["Mk/Mo/Yr OR Mk/Mo CVR"] = T1.apply(lambda x: 1 if x["Conversions_ad_group"]<11 and x["Conversions_Mk_Mo_Yr"]>10 else 0, axis = 1)
    
    # Step 1 - d
    T2 = T1.groupby(['Make Model'])[["Keyword_new", "Conversions", "Clicks"]].sum()
    T2 = T2.add_suffix('_Mk_Mo').reset_index()
    T2["Make Model"] = T2.apply(lambda x : x["Make Model"].strip(), axis = 1)
    T2["Group_Mk_Mo_CVR"] = T2.apply(lambda x : x["Conversions_Mk_Mo"]/x["Clicks_Mk_Mo"], axis = 1)
    
    T1 = pd.merge(T1, T2, left_on=T1['Make Model'], right_on = T2["Make Model"], how='left')
    T1 = T1.rename(columns={"Make Model_x": "Make Model"})
    T1 = T1.drop(["Make Model_y"], axis = 1)
    T1.reset_index()
    T1["New_KW_Bid"] = T1.apply(lambda x: x["Group_Mk_Mo_CVR"] * x["ASR"] if x["Conversions_Mk_Mo_Yr"]<11 and x["Conversions_Mk_Mo"]>=10 else x["New_KW_Bid"], axis = 1)
    T1["Mk/Mo/Yr OR Mk/Mo CVR"] = T1.apply(lambda x: 1 if x["Conversions_Mk_Mo_Yr"]<11 and x["Conversions_Mk_Mo"]>10 else x["Mk/Mo/Yr OR Mk/Mo CVR"], axis = 1)
    
    #T1.to_csv("Step_1_new_a.csv", index = 1)
    # Step 2 
    df_invt_current['combined']=df_invt_current.apply(lambda x:'%s%s%s' % (x['Make'],x['Model'],x['Year']),axis=1)
    df_invt_hist['combined_hist']=df_invt_hist.apply(lambda x:'%s%s%s' % (x['Make'],x['Model'],x['Year']),axis=1)
    
    # Joining T1 and df_invt_current ( current inventory sheet )
    T1 = T1.merge(df_invt_current, left_on='Keyword_new', right_on='combined', how='left')
    T1 = T1.drop(["Make", "Model", "Year_y", "Car_model", "Car_make"], axis = 1)
    
    # Joining T1 and df_invt_hist ( historical inventory sheet )
    T1 = T1.merge(df_invt_hist, left_on='Keyword_new', right_on='combined_hist', how='left')    
    T1 = T1.drop(["combined", "Year_x", "combined_hist", "Make", "Model", "Year" ], axis=1)
    T1["Reduction_Perc"] = 0
    T1["New_KW_Bid_2"] = "None"        
    
    # a) Adjust bid based on current onsite inventory
    # If current Mk/Mo/Yr inv < hist Mk/Mo/Yr inv
    # Reduce KW bid by % equal to half the % diff between current and historical inv
    # E.g., if hist avg is 20 and current inv is 15, reduce bid by 12.5% (i.e., half of 25%)
    
    T1["Reduction_Perc"] = np.where(T1["CurrentOnsiteInventory"] < T1["HistAvgInv"], (1-((T1["HistAvgInv"]-T1["CurrentOnsiteInventory"])/(2*T1["HistAvgInv"]))),0)
    T1["New_KW_Bid_2"] = np.where(T1["Reduction_Perc"] > 0, (T1["New_KW_Bid"] * T1["Reduction_Perc"]), T1["New_KW_Bid"])     
   # T1.to_csv("Step_2_new_a.csv", index = 1)           
    
    # b) Adjust bid based on Mkt CVR only for KWs whose bids were calculated based on Mk/Mo/Yr or Mk/Mo CVR 
    # (i.e., not based on KW or AG CVR)
    # Increase/decrease KW bid by the half the % above or below overall site CVR the market CVR is relative to overall site average
    # i.e., if overall CVR for the entire site is 1.0% and DAL overall CVR is 1.07%, increase bids for KWs in DAL by 3.5%
    
    overall_CVR = T1["Conversions"].sum()/T1["Clicks"].sum()
    T2 = T1.groupby(['Market'])[["Market", "Conversions", "Clicks"]].sum()
    T2 = T2.add_suffix('_Market').reset_index()
    T2["Market_CVR"] = T2.apply(lambda x : x["Conversions_Market"]/x["Clicks_Market"], axis = 1)
    
    T1 = pd.merge(T1, T2, left_on=T1['Market'], right_on = T2["Market"], how='left')
    T1 = T1.rename(columns={"Market_x": "Market"})
    T1 = T1.drop(["Market_y"], axis = 1)
    T1.reset_index()
    
    T1["New_KW_Bid_tmp"] = T1.apply(lambda x: 1 + (x["Market_CVR"]-overall_CVR)*100/2 if x["Market_CVR"] > overall_CVR  else (1 - (overall_CVR-x["Market_CVR"])*100/2), axis = 1)
    T1["New_KW_Bid_2"] = T1.apply(lambda x: x["New_KW_Bid_tmp"] if x["Mk/Mo/Yr OR Mk/Mo CVR"]==1 else x["New_KW_Bid_2"], axis = 1)
    T1 = T1.drop(["New_KW_Bid_tmp"], axis=1)
    #T1.to_csv("Step_1_new_b.csv", index = 1)
    
    # c) Cap bids at reasonable levels, based on their quality score
    # KWs with QS>7 cannot be higher than Est First Pos Bid
    # KWs with QS<8 and QS>5 cannot be higher than average of Est Top of Page Bid and Est First Pos Bid
    # KWs with QS<6 cannot be higher than (Est Top of Page Bid *0.9) + (Est First Pos Bid *0.1)
    # No bids can be higher than $12
    
    T1["New_KW_Bid_2"] = T1.apply(lambda x: x["Est First Pos. Bid"]  if x["Quality score"]>= 7 and x["New_KW_Bid_2"] > x["Est First Pos. Bid"] else x["New_KW_Bid_2"], axis = 1)
    T1["New_KW_Bid_2"] = T1.apply(lambda x: (x["Est First Pos. Bid"]+x["Est Top of Page Bid"])/2  if x["Quality score"]< 8 and x["Quality score"] >= 5 and x["New_KW_Bid_2"] > (x["Est First Pos. Bid"]+x["Est Top of Page Bid"])/2 else x["New_KW_Bid_2"], axis = 1)
    T1["New_KW_Bid_2"] = T1.apply(lambda x: (x["Est First Pos. Bid"]*.1 +x["Est Top of Page Bid"] *.9)  if x["Quality score"]<6 and x["New_KW_Bid_2"] > (x["Est First Pos. Bid"]*.1 +x["Est Top of Page Bid"] *.9) else x["New_KW_Bid_2"], axis = 1)
    T1["New_KW_Bid_2"] = T1.apply(lambda x: 12  if x["New_KW_Bid_2"] >12 else x["New_KW_Bid_2"], axis = 1)
    
    
    # d) Cap bids of broad match KWs
    # Ensure that no bid for a broad match KW is greater than any bid for an exact match KW within the same ad group
    # E.g., if bids for exact match KWs within the same ad group are $1.50, $1.75 and $1.60, 
    # then if a broad match KW with a calculated of bid of $2.00 should have its bid
    T2 = T1[T1["Match type"] == "Exact"]
    T3 = T2[["Ad group", "New_KW_Bid_2"]].groupby(['Ad group'])[["New_KW_Bid_2"]].min()
    T3 = T3.add_suffix('_Min').reset_index()
    
    T1 = pd.merge(T1, T3, left_on=T1['Ad group'], right_on = T3["Ad group"], how='left')
    T1 = T1.rename(columns={"Ad group_x": "Ad group"})
    T1 = T1.drop(["Ad group_y"], axis = 1)
    T1.reset_index()
    
    T1["New_KW_Bid_2"] = T1.apply(lambda x: x["New_KW_Bid_2_Min"] if x["Match type"] == "Broad" and x["New_KW_Bid_2"] > x["New_KW_Bid_2_Min"] else x["New_KW_Bid_2"], axis = 1)
       
    # Exporting the file as Carvana_KW_BID_Data_Report
    T1 = T1.rename(columns={"New_KW_Bid_2": "Final KW Bid"})
    T2= pd.DataFrame()
    T2["KW ID"] = T1["KW ID"]
    T2["Final KW Bid"] = T1["Final KW Bid"]
    del T1
    T2.to_csv("Carvana_KW_BID_Data_Report_NEW.csv", index = 0)       
    
# This is the starting point
if __name__ == "__main__":
    main()
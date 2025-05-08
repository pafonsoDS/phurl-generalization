import pandas as pd
import json
import sklearn
import numpy as np
import re

def clean_url(url): # important to remove unbalancing: ds had most stripped but if we keep a lot of www, if theres unbalancing among www there might be bias [xai print]
    url = re.sub(r"^[^a-zA-Z0-9]+", "", url)  # start by removing unwanted initial characters like quotes or spaces (otherwise www remains (36 in trainW))
    #remove common prefixes
    prefixes_to_remove = ['http://', 'https://', 'www.']
    for prefix in prefixes_to_remove:
        if url.startswith(prefix):
            url = url[len(prefix):]
    return url

def preprocess_df(df, prefix = True, lower = False): #add lower; cleanURL, add 2ID as bonus
    '''preprocess function -> takes care of URL prefix cleaning and lower() for generalization purposes'''
    if prefix:
        df['url'] = df['url'].apply(clean_url) #double clean
        df['url'] = df['url'].apply(clean_url)
    if lower:
        df['url'] = df['url'].str.lower()
    return df
################################################# URL NET #################################################

def process_and_save(df, save_path): 
    '''saves df to .txt as URLNet-friendly format
    save @ ds4urlnet/{ds_name}_formatted.txt
    '''
    # assume 'url' col for urls, and 'status' for labels
    
    # Map the labels to +1 and -1
    df['status'] = df['status'].map({0: '-1', 1: '+1'})

    print(df.head())
    print("\n")
    
    # Format the dataset as "<URL label><tab><URL string>"
    formatted_data = df[['status', 'url']].astype(str)
    formatted_data = formatted_data.apply(lambda x: f"{x['status']}\t{x['url']}", axis=1)
    
    #Save the formatted dataset
    with open(save_path, 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(formatted_data))

    print(f"Dataset saved to '{save_path}'.")


def process_and_save_json(df, save_path): 
    '''saves df to .json in a URLNet-friendly format'''
     
    # assume 'url' col for urls, and 'status' for labels
    
    # Map the labels to +1 and -1
    df['status'] = df['status'].map({0: -1, 1: 1}) #this ensures numeric values for JSON compatibility

    print(df.head())
    print("\n")
    
    #convert the DataFrame to a list of dictionaries
    formatted_data = df.to_dict(orient='records')
    
    #save the formatted dataset as JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Dataset saved to '{save_path}'.")

##############################################################################################################################################
def buildDataframes(source=False):
    ################
    ### kaggle4s ###
    ################
    kaggle4s = pd.read_csv('kaggle4s/url_kaggle.csv')
    kaggle4s['status'] = 1-kaggle4s['status'] # no additional prep because this was the base struct with 'url', 'status'
    if source: kaggle4s['source']=0
    print("Kaggle4S sucessfully loaded!")
    ##################
    ### kaggle2020 ###
    ##################
    kaggle2020 = pd.read_csv("kaggleIndianURL-102/phishing_site_urls.csv")
    kaggle2020 = kaggle2020.rename(columns={"Label": "status", "URL": "url"})
    kaggle2020['status'] = kaggle2020['status'].replace({'bad': 1, 'good': 0})
    if source: kaggle2020['source']=1
    print("Kaggle2020 sucessfully loaded!")
    ####################
    #### kaggle2019 ####
    ####################
    kaggle2019 = pd.read_csv('kaggleSiddarth2019-70/urldata.csv')
    kaggle2019 = kaggle2019.rename(columns={'label': 'status'})
    kaggle2019['status'] = kaggle2019['result']
    kaggle2019 = kaggle2019.drop(columns=['Unnamed: 0','result'])
    if source: kaggle2019['source']=2
    print("Kaggle2019 sucessfully loaded!")
    ##################
    ### kaggle2021 ###
    ##################
    kaggle2021 = pd.read_csv('kaggleManu2021-46/malicious_phish.csv')
    kaggle2021 = kaggle2021.rename(columns={'type': 'status'})
    kaggle2021['status'] = kaggle2021['status'].apply(lambda x: 0 if x == 'benign' else 1)
    if source: kaggle2021['source']=3
    print("Kaggle2021 sucessfully loaded!")
    #############
    #### LBL ####
    #############
    lbl = pd.read_excel('lookbe4leap/URL.xlsx')
    lbl = lbl.rename(columns={"Category": "status", "Data": "url"})
    lbl['status'] = lbl['status'].replace({'spam': 1, 'ham': 0})
    if source: lbl['source']=4
    print("Look Before Leap data sucessfully loaded!")
    #################
    ### EBBU-2017 ###
    #################
    path_base = 'sahingoz_url/'
    path_legit = path_base + 'sahingoz_legitimate_36400.json'
    path_phish = path_base + 'sahingoz_phishing_37175.json'
    with open(path_legit, 'r') as file:
        sh_legit = json.load(file)
        for i,el in enumerate(sh_legit):
            sh_legit[i] = [el,0]
        urls_0 = pd.DataFrame(sh_legit, columns=['url', 'status'])

    with open(path_phish, 'r') as file:
        sh_phish = json.load(file)
        for i,el in enumerate(sh_phish):
            sh_phish[i] = [el,1]
        urls_1 = pd.DataFrame(sh_phish, columns=['url','status'])

    ebbu = pd.concat([urls_0, urls_1], ignore_index=True)
    #shuffle final result
    ebbu = sklearn.utils.shuffle(ebbu, random_state=42)  
    ebbu = ebbu.reset_index(drop = True)
    if source: ebbu['source']=5
    print("EBBU2017 sucessfully loaded!")
    ##############
    ### HISPAR ###
    ##############
    hispar_train = pd.read_csv('hispar/hp_train.csv')
    hispar_test = pd.read_csv('hispar/hp_test.csv')
    hispar_val = pd.read_csv('hispar/hp_valid.csv')

    hispar = pd.concat([hispar_train, hispar_test, hispar_val])
    hispar = hispar.drop(columns=['Unnamed: 0.1', 'index', 'Unnamed: 0'])

    hispar = hispar.rename(columns={'label': 'status'})
    if source: hispar['source']=6
    print("HISPAR sucessfully loaded!")
    ##############
    #### ISCX ####
    ##############
    iscx_train = pd.read_csv('iscx_url/iscx_train.csv')
    iscx_test = pd.read_csv('iscx_url/iscx_test.csv')
    iscx_val = pd.read_csv('iscx_url/iscx_valid.csv')

    iscx = pd.concat([iscx_train, iscx_test, iscx_val])
    iscx = iscx.drop(columns=['index', 'Unnamed: 0.1', 'Unnamed: 0'])
    iscx = iscx.rename(columns={'label':'status'})
    if source: iscx['source']=7
    print("ISCX-URL2016 sucessfully loaded!")

    dfs = {
        'kaggle4s': kaggle4s,
        'kaggle2019': kaggle2019,
        'kaggle2020': kaggle2020,
        'kaggle2021': kaggle2021,
        'lbl': lbl,
        'ebbu': ebbu,
        'hispar': hispar,
        'iscx': iscx,
    }
    return dfs
################################################################################################################################################
def buildPhish():
    '''returns openphish, phishtank dfs (only left check dups, below)'''
    openphish = pd.read_csv("openphish/openphish6306.csv", header=None)
    openphish.columns = ['url']
    openphish['status'] = 1

    openphish = preprocess_df(openphish)

    with open("phishtank/phish-tank.json", 'r') as file:
        json_phtank = json.load(file)
    phishtank = pd.DataFrame(json_phtank)[['url']]
    phishtank['status'] = 1

    phishtank = preprocess_df(phishtank)

    return openphish, phishtank
########################################################################
def remove_common_urls(df_val, df_ultimate):
    """
    Removes rows from df_phishtank where the 'url' column matches any value in the 'url' column of df_ultimate.
    
    Args:
        df_val (pd.DataFrame): Validation dataset.
        df_ultimate (pd.DataFrame): Dataset containing URLs to exclude.
        
    Returns:
        pd.DataFrame: Filtered df_val.
    """
    filtered_df = df_val[~df_val['url'].isin(df_ultimate['url'])]
    return filtered_df.reset_index(drop=True) #to prevent keyErrors in pytorch
########################################################################
def destroy_DC(df, verbose=False):
    """
    Remove all rows corresponding to any URL that occurs more than once in the dataframe.
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least a 'url' column.
        verbose (bool): If True, print out the list of duplicate URLs that will be removed.
        
    Returns:
        pd.DataFrame: A copy of df with ALL duplicate URLs removed.
    """
    # Count occurrences of each URL
    url_counts = df['url'].value_counts()
    
    # Identify URLs that appear more than once
    duplicate_urls = url_counts[url_counts > 1].index.tolist()
    
    if verbose:
        print("Found duplicate URLs (to be removed):")
        for url in duplicate_urls:
            print(f" - {url}")
    
    # Remove any row whose URL is in the list of duplicate URLs
    df_cleaned = df[~df['url'].isin(duplicate_urls)].copy()
    
    return df_cleaned
########################################################################
def resolve_duplicates(df, count = False, verbose = False): #destroy CONFLICTS
    """
    Resolves duplicate URLs with conflicting labels by applying a priority rule.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'url' and 'status'.
        resolving strategy: Remove URLs with conflicting labels.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates resolved.
    """
    conflict_counts = df.groupby('url')['status'].nunique() #count unique labels for each URL -> if ok 1, if conflicted 2. then we simply count (below) the ones with conflict_counts == 2 (>1)
    
    conflicting_urls = conflict_counts[conflict_counts > 1].index #identify conflicting URLs (those with more than one unique label!! [we could put conflict_counts == 2])
    #print(f"conflicting_urls = {conflicting_urls}")
    l1 = len(df)
    df = df[~df['url'].isin(conflicting_urls)]#remove conflicting URLs!!!
    l2 = len(df)

    if verbose: print(f"There were {l1-l2} conflicting URLs that were removed.") #careful. ~(1/2)*conflicts in each ds (ex: Df_A, Df_B, l(A)>>l(B); 40k conflicts, 20k ea, if 40k=l(B) then conf 1 [F]) - thats why unique in "urlDup.ipynb"
    if count: return l1-l2
    
    return df
###add Discarder and n_combine #and take things out if needed
def discarder(df, verbose = False): #DESTROY C, remove dups (keep 1 entry)
    df_no_conflicts = resolve_duplicates(df)
    df_discarded_bad = df_no_conflicts.drop_duplicates(subset=['url'])
    if verbose == True:
        print(f"There were {len(df)-len(df_discarded_bad)} duplicates/conflicts removed.")
    return df_discarded_bad
###############################################################
def n_combiner(datasets, subset = None, prep = False, to_lower = False, pref = False, verbose = False): #changed all vars to false, maybe change source n_combiner script to explicitly set var = T
    '''this function allows us to choose the datasets we want to merge and discards non relevant URLs
    notes: 
        - a URL is discarded if he is a duplicate (only original ENTRY is KEPT) or if he conflicts with another URL (no majority voting for label not risking false negative)
        - subset is an array containing a subset of ['kaggle4s', 'kaggle2020', 'kaggle2019', 'kaggle2021', 'lbl', 'ebbu', 'hispar', 'iscx'] '''

    if subset == None: 
        #assume 8 fold
        subset = ['kaggle4s', 'kaggle2020', 'kaggle2019', 'kaggle2021', 'lbl', 'ebbu', 'hispar', 'iscx']

    if subset != None:
        datasets_new = {}
        for name in subset:
            datasets_new[name] = datasets[name]
        datasets = datasets_new
    
    subset_str = f"Starting merge for subset = {subset}"
    print(subset_str)
    ################################ concat ################################
    arr_df = []
    for ds_name in datasets:
        arr_df.append(datasets[ds_name])
    
    combined_df = pd.concat(arr_df, axis=0, ignore_index=True) ###very important the order of arr_df, the first url in considered "king", others dupped drop (URLs are the same, just different source) 

    if verbose:
        sep_length = len(subset_str)+3  # +3 -> slightly cleaner
        sep = '-' * sep_length
        print(sep)
        print(f"N_initial = {len(combined_df)}")

    if prep:
        combined_df = preprocess_df(combined_df, lower=to_lower, prefix=pref)
    ################################ discarding ######################################
    df_discard = discarder(combined_df)
    nr_discard = len(combined_df)-len(df_discard)
    if verbose: 
        print(f"=> Found {nr_discard} URLs to discard.")
        print(f"N_final = {len(df_discard)}")
        print("")

    print(f"Dataframe w/ subset {subset} sucessfully built.")
    return df_discard
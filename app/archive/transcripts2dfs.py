#%%

import os
import pandas as pd
import numpy as np
from datetime import timedelta

input_path = "../data/transcripciones/originales/"
output_path = "../data/transcripciones/limpias/"

#%%

# loop over all files in the input directory
all_files = os.listdir(input_path)
all_files = [f for f in all_files if f.endswith(".tsv")]

# # select only the files that are not already in the output directory
# processed_files = os.listdir(output_path)
# processed_files = [f.replace(".csv", "") for f in processed_files if f.endswith(".csv")]

# all_files = [f for f in all_files if f.replace(".tsv", "") not in processed_files]

print(f"Number of files to process: {len(all_files)}")

#%%

# min duration in seconds
min_duration = 30
for file_name in all_files:
    print(f"Processing file {file_name}\n")

    df_docs = pd.read_csv(f'{input_path}{file_name}', sep='\t')

    # replace nan with "INAUDIBLE"
    df_docs = df_docs.replace(np.nan, 'INAUDIBLE', regex=True)

    # convert start and end to seconds
    df_docs['start'] = df_docs['start'].apply(lambda x: round(x/1000))
    df_docs['end'] = df_docs['end'].apply(lambda x: round(x/1000))

    # convert seconds to HH:MM:SS format
    df_docs['start'] = df_docs['start'].apply(lambda x: str(timedelta(seconds=x)))
    df_docs['end'] = df_docs['end'].apply(lambda x: str(timedelta(seconds=x)))

    # de-duplicate any text that appears in more than two consecutive rows
    for i in range(1, len(df_docs)-1):
        if df_docs.iloc[i-1]['text'] == df_docs.iloc[i]['text'] == df_docs.iloc[i+1]['text']:
            #print(df_docs.iloc[i-1]['text'])
            df_docs.iloc[i]['text'] = ''

    # drop empty rows
    df_docs = df_docs[df_docs['text'] != '']
    df_docs

    # transform start and end columns to HH:MM:SS date format
    df_docs['start'] = pd.to_datetime(df_docs['start'], format='%H:%M:%S')
    df_docs['end'] = pd.to_datetime(df_docs['end'], format='%H:%M:%S')
    df_docs

    # concatenate rows together 
    durations = []
    text = []

    current_i = 0
    while current_i < len(df_docs):
        
        block_ready = False
        step = 0

        while not block_ready:
            
            if current_i + step >= len(df_docs):
                # This is the last possible 'end' value in the DataFrame
                current_duration = df_docs.loc[len(df_docs)-1, "end"] - df_docs.loc[current_i, "start"]
                block_ready = True
            else:
                current_duration = df_docs.loc[current_i + step, "end"] - df_docs.loc[current_i, "start"]
            
            current_duration = current_duration.total_seconds()
            if current_duration <  min_duration:
                step += 1
            else:
                block_ready = True

        # Make sure that the end index does not exceed the length of the DataFrame
        end_index = min(current_i + step, len(df_docs)-1)
            
        durations.append((str(df_docs.loc[current_i, "start"].time()), str(df_docs.loc[end_index, "end"].time())))
        block_text = ""
        for j in range(current_i, end_index + 1):
            block_text += df_docs.loc[j, "text"] + " "
        text.append(block_text)
        
        current_i += step + 1

    # consolidate into a new dataframe
    df_clean = pd.DataFrame({"duration": durations, "text": text})
    # add the name of the file
    df_clean["file_name"] = file_name.replace(".tsv", "")

    # save dataframe to csv
    df_clean.to_csv(output_path + file_name.replace(".tsv", ".csv"), index=False)

    # save text to txt file line by line
    with open(f'{output_path}{file_name.replace(".tsv", ".txt")}', 'w') as f:
        for _, row in df_clean.iterrows():
            # add start and end
            f.write(f"[{row['duration'][0]} - {row['duration'][1]}]: ")
            f.write(row['text'] + "\n\n")

#%%

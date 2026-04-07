import os
import imghdr
import pandas as pd

def fix_image_extensions(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip non-files
        if not os.path.isfile(file_path):
            continue

        # Detect image type
        img_type = imghdr.what(file_path)

        if img_type is None:
            print(f"Not an image: {filename}")
            continue

        # Normalize jpeg to jpg
        if img_type == 'jpeg':
            img_type = 'jpg'

        # Get current extension
        current_ext = os.path.splitext(filename)[1][1:].lower()

        if current_ext != img_type:
            new_filename = os.path.splitext(filename)[0] + f".{img_type}"
            new_path = os.path.join(folder_path, new_filename)
            os.rename(file_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")
        else:
            print(f"Already correct: {filename}")


def combine_tags(df):
    """
        Combine misleading and not misleading tags in two columns to make it more clear 
    """
    def combine_tags(row, cols):
        return [col for col in cols if row.get(col) == 1]

    misleading_cols = [
        "misleadingOther", "misleadingFactualError", "misleadingManipulatedMedia", 
        "misleadingOutdatedInformation", "misleadingMissingImportantContext", 
        "misleadingUnverifiedClaimAsFact", "misleadingSatire"
    ]

    not_misleading_cols = [
        "notMisleadingOther", "notMisleadingFactuallyCorrect", 
        "notMisleadingOutdatedButNotWhenWritten", "notMisleadingClearlySatire", 
        "notMisleadingPersonalOpinion"
    ]

    df["misleadingTags"] = df.apply(lambda row: combine_tags(row, misleading_cols), axis=1)
    df["notMisleadingTags"] = df.apply(lambda row: combine_tags(row, not_misleading_cols), axis=1)

    df.drop(columns=misleading_cols + not_misleading_cols, inplace=True)

    return df


def read_prepare(file):
    """
        Read CSV files and prepare their columns
    """ 

    df1 = pd.read_csv(f"{file}_tweets.csv")
    df2 = pd.read_csv(f"{file}.csv")

    df1 = df1.rename(columns={
        "iteration_id": "id",
        "tweet_id": "tweetId",
        "text": "tweetText",
        "created_at_datetime": "tweetCreatedAt",
        "user": "tweetUser"
    })
    df2 = df2.rename(columns={
        "noteAuthorParticipantId": "noteUser",
        "createdAtMillis": "noteCreatedAt",
        "summary": "noteText"
    })

    return df1,df2

def class_merge_and_process(file):
    """
        Gathers note data and tweet data and combines them to form each class' df
    """

    print(f"\nProcessing file: {file}")
    
    df1,df2 = read_prepare(file)

    # Drop tweets with empty or missing text/media
    empty_text_count = (df1['tweetText'].str.strip() == '').sum()
    print(f"Rows with empty 'text': {empty_text_count}")
    df1 = df1[df1['tweetMediaFiles'].notna() & df1['tweetText'].notna()]

    # Drop duplicates
    df1 = df1.drop_duplicates(subset="tweetId")
    df2 = df2.drop_duplicates(subset="tweetId")

    # Select columns to keep
    df1 = df1[["id", "tweetId", "tweetUrl", "tweetText", "tweetMediaFiles","tweetUser","tweetCreatedAt"]]
    df2 = df2[[
        "tweetId", "noteId", "noteUser", "noteCreatedAt", "classification", "trustworthySources",
        "noteText", "isMediaNote",
        "misleadingOther", "misleadingFactualError", "misleadingManipulatedMedia", 
        "misleadingOutdatedInformation", "misleadingMissingImportantContext", 
        "misleadingUnverifiedClaimAsFact", "misleadingSatire",
        "notMisleadingOther", "notMisleadingFactuallyCorrect", "notMisleadingOutdatedButNotWhenWritten",
        "notMisleadingClearlySatire", "notMisleadingPersonalOpinion"
    ]]

    # Combine tag columns in df2
    df2 = combine_tags(df2)

    # Merge the dataframes
    merged_df = pd.merge(df1, df2, on="tweetId", how="inner")
    merged_df['class'] = file
    
    merged_df['uniqueID'] = file[:2] + merged_df['id'].astype(str).str.zfill(4)

    return merged_df

def process_and_build_dataset(*files, output_path):
    """
        Processes each class and combines them to final dataset
    """
    merged_dfs = []

    for file in files:
        df = class_merge_and_process(file)  
        merged_dfs.append(df)

    combined_df = pd.concat(merged_dfs, ignore_index=True)

    # Drop id column as we will use uniqueID
    combined_df = combined_df.drop(columns=['id'])

    # Define column order
    column_order = [
        'uniqueID', 'noteText', 'tweetText', 'tweetUrl','tweetMediaFiles', 'classification', 'tweetId', "tweetUser","tweetCreatedAt",'noteId','noteUser', 'noteCreatedAt','isMediaNote', 'trustworthySources', 'misleadingTags', 'notMisleadingTags', 'class'
    ]
    combined_df = combined_df[[col for col in column_order]]

    combined_df.to_csv(output_path, index=False)
    print(f"Final combined dataset saved as {output_path}")



# Example run
# files = ['factually_correct', 'miscaptioned']
# dataset_path = "dataset.csv"

# process_and_build_dataset(*files, output_path=dataset_path)

# fix_image_extensions(dataset_path)



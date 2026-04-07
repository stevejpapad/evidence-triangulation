import os
import csv
import time
import asyncio
import pandas as pd
import httpx, httpcore
from twikit import Client
from twikit.guest import GuestClient

# Common transient errors httpx/httpcore
RETRYABLE_ACTIVATE_ERRORS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpcore.ConnectTimeout,
)

# Constants
BATCH_SIZE = 145
DELAY_BETWEEN_TWEETS = 3 


# Helper to avoid breaking when user cant be reactivated
async def activate_guest_simple(client, retries=10, sleep_seconds=60):
    """
    Re-activate GuestClient with simple retries.
    - On failure: sleep `sleep_seconds` and retry.
    Returns True on success, False if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            await client.activate()
            print("Guest client activated")
            return True
        except RETRYABLE_ACTIVATE_ERRORS as e:
            print(f"activate() timeout/network issue ({type(e).__name__}) — attempt {attempt}/{retries}. Sleeping {sleep_seconds}s…")
            await asyncio.sleep(sleep_seconds)
        except Exception as e:
            print(f"activate() failed (non-retryable): {type(e).__name__}: {e}")
            return False
    print("activate() failed after all retries")
    return False

async def get_tweets_by_ids(input_path, output_path,image_dir, mode, start_id=None, end_id=None):

    # Load tweet IDs from CSV
    df = pd.read_csv(input_path)
    tweet_ids = df["tweetId"].dropna().astype(str).tolist()

    # Create image dir
    os.makedirs(image_dir, exist_ok=True)

    
    # Run for a subset
    # Normalize None → full range
    if start_id is None and end_id is None:
        start_id = 0
        end_id = len(tweet_ids)

    # Now slice 
    tweet_ids = tweet_ids[start_id:end_id]


    """
        Connection to x.com with Twikit
        
        A) Using credentials
         - The first time, log in using the login method and save cookies   
         - After the second time, load the saved cookies (to avoid bans)

        B) Connect as a Guest- Easier, no account bans

    """

    if mode == 'login':
        client = Client()
        client.load_cookies('cookies.json')

    elif mode == 'guest': 
        client = GuestClient()
        await client.activate() 
    else:
        print("Invalid mode. Must be 'login' or 'guest'")
        return

    results = []
    wrote_main_header = False
    
    for i in range(0, len(tweet_ids), BATCH_SIZE):
       batch = tweet_ids[i:i + BATCH_SIZE]
       print(f"\nProcessing batch {i//BATCH_SIZE + 1} ({len(batch)} tweets)...")

       results = [] 

       for idx, tweet_id in enumerate(batch):
            try:
                tweet = await client.get_tweet_by_id(tweet_id)
                
                if tweet is None:
                    raise ValueError(f"\n {i + idx + start_id + 1}) Tweet {tweet_id} is None (possibly deleted). Skipping.")
    
                media_files = []
                if tweet.media:
                    for media_idx, media in enumerate(tweet.media):
                        if media.type == 'photo':
                            # Save with iteration ID
                            filename = f"{image_dir}/photo_{i + idx + start_id + 1}_{media_idx}.jpg"
                            await media.download(filename)
                            # Store relative paths for downstream (relative to images/ root)
                            media_files.append(filename.replace("images/", ""))

                result = {
                    "iteration_id": i + idx + start_id + 1, 
                    "tweet_id": tweet_id,
                    "created_at_datetime": tweet.created_at_datetime if mode == 'login' else tweet.created_at,
                    "user": tweet.user.name if tweet.user else None,
                    "user_followers": tweet.user.followers_count if tweet.user else None,
                    "text": tweet.text,
                    "lang": tweet.lang,
                    "in_reply_to_id": tweet.in_reply_to,
                    "in_reply_to_url": ("https://x.com/i/web/status/" + tweet.in_reply_to) if tweet.in_reply_to else None,
                    "quoted_post_id": tweet.quote.id if tweet.quote else None,
                    "quoted_post_text": tweet.quote.full_text if tweet.quote else None,
                    "retweeted_tweet": tweet.retweeted_tweet.full_text if tweet.retweeted_tweet else None,
                    "tweetMediaFiles": ", ".join(media_files) if media_files else None,
                    "reply_count": tweet.reply_count,
                    "favorite_count": tweet.favorite_count,
                    "view_count": tweet.view_count,
                    "view_count_state": tweet.view_count_state,
                    "retweet_count": tweet.retweet_count,
                    "bookmark_count": tweet.bookmark_count,
                    "place": tweet.place.full_name if hasattr(tweet, "place") and tweet.place else None,
                    "replies": len(tweet.replies) if hasattr(tweet, "replies") and tweet.replies else None,
                    "hashtags": ", ".join(tweet.hashtags) if tweet.hashtags else None,
                    "thumbnail_title": tweet.thumbnail_title,
                    "thumbnail_url": tweet.thumbnail_url,
                    "urls": ", ".join([u['expanded_url'] for u in tweet.urls]) if tweet.urls else None,
                    "full_text": tweet.full_text,
                    'tweetUrl':f"https://x.com/i/web/status/{tweet_id}",
                }

                results.append(result)

                print(f"{i + idx + start_id + 1}) Tweet {tweet_id} processed.")
                await asyncio.sleep(DELAY_BETWEEN_TWEETS)


            except Exception as e:
                print(f"{i + idx + start_id + 1}) Error processing tweet {tweet_id}: {e}")

                results.append({
                    "iteration_id": i + idx + start_id + 1,
                    "tweet_id": tweet_id,
                    "created_at_datetime": None,
                    "user": None,
                    "user_followers": None,
                    "text": None,
                    "lang": None,
                    "in_reply_to_id": None,
                    "in_reply_to_url": None,
                    "quoted_post_id": None,
                    "quoted_post_text": None,
                    "retweeted_tweet": None,
                    "tweetMediaFiles": None,
                    "reply_count": None,
                    "favorite_count": None,
                    "view_count": None,
                    "view_count_state": None,
                    "retweet_count": None,
                    "bookmark_count": None,
                    "place": None,
                    "replies": None,
                    "hashtags": None,
                    "thumbnail_title": None,
                    "thumbnail_url": None,
                    "urls": None,
                    "full_text": None,
                    'tweetUrl':f"https://x.com/i/web/status/{tweet_id}",
                })

                await asyncio.sleep(DELAY_BETWEEN_TWEETS)


            if mode == 'guest':
                if (i + idx + start_id + 1) % 40 == 0:
                    print("Re-activating guest client to refresh token...")
                    ok = await activate_guest_simple(client, retries=10, sleep_seconds=60)
                    if not ok:
                        print("Could not re-activate after retries — continuing (you may see failures until next attempt).")


       # Save the current batch to a csv
       first_iter_id = i + start_id + 1
       last_iter_id = i + len(batch) + start_id
       batch_output_path = output_path.replace(".csv", f"_{first_iter_id}-{last_iter_id}.csv")

       print(f"Saving batch to {batch_output_path}...")

       keys = results[0].keys() if results else []
       with open(batch_output_path, mode='w', newline='', encoding='utf-8') as f:
           writer = csv.DictWriter(f, fieldnames=keys)
           writer.writeheader()
           writer.writerows(results)

       # Also write/append into the main output_path that downstream expects
       if results:
           with open(output_path, mode='a', newline='', encoding='utf-8') as f:
               writer = csv.DictWriter(f, fieldnames=keys)
               if not wrote_main_header:
                   writer.writeheader()
                   wrote_main_header = True
               writer.writerows(results)


       # Wait between batches if more tweets remain
       if i + BATCH_SIZE < len(tweet_ids):
           print(f"\n Waiting 5 minutes to respect rate limits...")
           for t in range(5, 0, -1):
               print(f"  ... {t} minutes left")
               time.sleep(61)




# Example run

# START_ID = 0
# END_ID =   10 
# MODE = 'guest'

# input_path = f'/input/path.csv'
# output_path = f'output/path.csv'
# image_dir = f"image/dir"

# asyncio.run(get_tweets_by_ids(input_path, output_path,image_dir, MODE, START_ID,END_ID))







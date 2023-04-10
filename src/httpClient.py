import requests
import datetime
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import json


class HttpClient:
    # BASE_URL = "https://oauth.reddit.com"
    # URL = f"{BASE_URL}/r"
    BASE_URL = "https://api.pushshift.io/reddit/search"
    POST_URL = f"{BASE_URL}/submission"
    POST_COMMENTS_URL = f"https://api.pushshift.io/reddit/submission/comment_ids"
    COMMENT_URL = f"{BASE_URL}/comment"

    headers = {'User-Agent': 'windows:study-project-echo-chambers-script:v0.0.1/u/schadenfreude2030'}

    def __init__(self):
        pass

    def fetch_submissions(self, subreddit, start_date, end_date):
        submissions = DataFrame([])
        counter = 0
        latest_created_utc = 0
        # get submissions
        print("fetching submissions")
        while counter < 10000:
            counter += 1
            if counter % 5 == 0:
                print(f"loop no {counter}. Items so far: {len(submissions)}")
            data = {
                "size": 500,
                "subreddit": subreddit,
                "before": int(end_date.replace(tzinfo=datetime.timezone.utc).timestamp()),
                "after": max(
                    int(start_date.replace(tzinfo=datetime.timezone.utc).timestamp()),
                    latest_created_utc
                ),
            }
            response = requests.get(self.POST_URL, params=data, headers=self.headers)
            if response.status_code in [419, 502, 504, 524]:
                time.sleep(5)
                continue
            if not response.ok:
                print("not ok")
                time.sleep(5)
                continue
            try:
                response_data = response.json()["data"]
                response_data = DataFrame.from_dict(response_data)
                submissions = pd.concat([submissions, response_data], ignore_index=True)
                if len(response_data) < 220:
                    break
                latest_created_utc = response_data.iloc[-1]["created_utc"]
            except json.decoder.JSONDecodeError:
                print("error")
            time.sleep(0.2)
        # get comments
        comments = DataFrame([])
        counter = 0
        latest_created_utc = 0
        print("fetching comments")
        while counter < 100000:
            counter += 1
            if counter % 5 == 0:
                print(f"loop no {counter}. Items so far: {len(comments)}")
            data = {
                "size": 500,
                "subreddit": subreddit,
                "before": int(end_date.replace(tzinfo=datetime.timezone.utc).timestamp()),
                "after": max(
                    int(start_date.replace(tzinfo=datetime.timezone.utc).timestamp()),
                    latest_created_utc
                ),
            }

            response = requests.get(self.COMMENT_URL, params=data, headers=self.headers)
            if response.status_code in [419, 502, 504, 524]:
                time.sleep(5)
                continue
            if not response.ok:
                print("not ok")
                time.sleep(5)
                continue
            try:
                response_data = response.json()["data"]
                response_data = DataFrame.from_dict(response_data)
                comments = pd.concat([comments, response_data], ignore_index=True)
                if len(response_data) < 220:
                    break
                latest_created_utc = response_data.iloc[-1]["created_utc"]
            except json.decoder.JSONDecodeError:
                print("error")

            time.sleep(0.2)
        #comments = comments.groupby("link_id")

        #def add_comments(row):
        #    if row.id in comments.groups:
        #        row.comments = comments.get_group(row.id)
        #    else:
        #        row.comments = []
        #    return row

        #submissions = submissions.apply(add_comments, axis=1)
        df_merged = pd.concat([
            submissions[["author", "author_fullname", "score"]],
            comments[["author", "author_fullname", "score"]]
        ], ignore_index=True)
        df_merged = df_merged[df_merged["author_fullname"].notna()]
        return df_merged

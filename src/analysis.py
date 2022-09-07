from typing import Dict, List
import pandas as pd
import json
import os.path


class User:
    LEANING_THRESHOLD = 0.5
    POSTS_TO_BE_RELEVANT = 5

    def __init__(self, name: str):
        self.id = name
        self.amount_posts_pro = 0
        self.amount_posts_con = 0
        self.votes_received_pro = 0
        self.votes_received_con = 0
        self.amount_downvoted_pro = 0
        self.amount_downvoted_con = 0

    def add_content(self, score: int, pro: bool = True):
        if pro:
            self.amount_posts_pro += 1
            self.votes_received_pro += score
            if score < 0:
                self.amount_downvoted_pro += 1
        else:
            self.amount_posts_con += 1
            self.votes_received_con += score
            if score < 0:
                self.amount_downvoted_con += 1

    def get_leaning(self):
        without_votes = (self.amount_posts_pro - self.amount_posts_con) / (
                self.amount_posts_pro + self.amount_posts_con)
        if abs(without_votes) > self.LEANING_THRESHOLD:
            return round(without_votes)
        # edge case: No votes received at all
        if self.votes_received_pro + self.votes_received_con == 0:
            return 0
        with_votes = (self.votes_received_pro - self.votes_received_con) / (
                self.votes_received_pro + self.votes_received_con)
        if abs(with_votes) > self.LEANING_THRESHOLD:
            return round(with_votes)
        return 0

    def is_relevant(self):
        return (self.amount_posts_pro + self.amount_posts_con) > self.POSTS_TO_BE_RELEVANT

    def get_metrics(self):
        metrics = {
            "posted_both": self.amount_posts_pro > 0 and self.amount_posts_con > 0,
            "amount_posts_pro": self.amount_posts_pro,
            "amount_posts_con": self.amount_posts_con,
            "average_score": (self.votes_received_pro + self.votes_received_con) /
                             (self.amount_posts_pro + self.amount_posts_con),
            "average_score_pro": self.votes_received_pro / self.amount_posts_pro
            if self.amount_posts_pro > 0 else 0,
            "average_score_con": self.votes_received_con / self.amount_posts_con
            if self.amount_posts_con > 0 else 0,
            "amount_downvoted_pro": self.amount_downvoted_pro,
            "amount_downvoted_con": self.amount_downvoted_con,
            "leaning": self.get_leaning()
        }

        return pd.Series(data=metrics)

    def to_json(self):
        return json.dumps({
            "id": self.id,
            "amount_posts_pro": self.amount_posts_pro,
            "amount_posts_con": self.amount_posts_con,
            "votes_received_pro": self.votes_received_pro,
            "votes_received_con": self.votes_received_con,
            "amount_downvoted_pro": self.amount_downvoted_pro,
            "amount_downvoted_con": self.amount_downvoted_con
        })

    def from_json(self, json_str):
        dict = json.loads(json_str)
        self.id = dict["id"]
        self.votes_received_pro = dict["votes_received_pro"]
        self.votes_received_con = dict["votes_received_con"]
        self.amount_posts_pro = dict["amount_posts_pro"]
        self.amount_posts_con = dict["amount_posts_con"]
        self.amount_downvoted_pro = dict["amount_downvoted_pro"]
        self.amount_downvoted_con = dict["amount_downvoted_con"]
        return self


class UserNetwork:

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.pro_users: List[User] = []
        self.con_users: List[User] = []
        self.undefined_users: List[User] = []

    def save(self, filepath):
        with open(filepath, "w+") as file:
            file.write(self.to_json())
            file.close()
        return self

    def load(self, filepath):
        with open(filepath) as file:
            content = json.load(file)
            file.close()
            self.from_json(content)
        return self

    def add_content(self, user_name: str, score: int, pro: bool = True):
        if user_name not in self.users:
            self.users[user_name] = User(user_name)
        self.users[user_name].add_content(score, pro)
        return self

    def get_metrics(self):
        df = pd.DataFrame([user.get_metrics() for user in self.users.values()])
        return df

    def __build_overview(self):
        for user in self.users.values():
            leaning = user.get_leaning()
            if leaning == 1:
                self.pro_users.append(user)
            if leaning == -1:
                self.con_users.append(user)
            if leaning == 0:
                self.undefined_users.append(user)

    def overview(self):
        self.__build_overview()
        return {
            "pro": self.pro_users,
            "relevant_pro": [user.is_relevant() for user in self.pro_users],
            "con": self.con_users,
            "relevant_con": [user.is_relevant() for user in self.con_users],
            "undefined": self.undefined_users,
            "relevant_undefined": [user.is_relevant() for user in self.undefined_users],
        }

    def __str__(self):
        overview = self.overview()
        return f"Pro users: {len(overview.get('pro'))} (relevant: {len(overview.get('relevant_pro'))})\n" \
               f"Con users: {len(overview.get('con'))} (relevant: {len(overview.get('relevant_con'))})\n" \
               f"Undefined users: {len(overview.get('undefined'))} (relevant: " \
               f"{len(overview.get('relevant_undefined'))}\n" \
               f"Total: {len(self.users)}"

    def to_json(self):
        return json.dumps({key: value.to_json() for key, value in self.users.items()})

    def from_json(self, json_dict):
        self.users = {}
        for key, value in json_dict.items():
            self.users[key] = User(key).from_json(value)
        return self


class Submission:

    def __init__(self, author: str, score: int):
        self.author = author
        self.score = score
        self.comments: [Submission] = []

    def add_comment(self, comment):
        self.comments.append(comment)

    def to_json(self):
        return json.dumps({
            "author": self.author,
            "score": self.score,
            "comments": [
                comment.to_json() for comment in self.comments
            ]
        })

    def from_json(self, json_str):
        dict = json.loads(json_str)
        self.author = dict.author
        self.score = dict.score
        self.comments = dict.comments
        return self


class SubredditHistory:

    def __init__(self, subreddit: str):
        self.subreddit: str = subreddit
        self.submissions: [Submission] = []

    def add_submission(self, submission: Submission, comments: [Submission]):
        for comment in comments:
            submission.add_comment(comment)
        self.submissions.append(submission)

    def to_json(self):
        return json.dumps([submission.to_json() for submission in self.submissions])

    def from_json(self, json_dict):
        self.submissions = []
        for item in json_dict:
            # TODO testen wie das gespeichert wird bevor ich es lade??
            self.submissions.append()


def build_network_leaning(data: pd.DataFrame, filepath: str, pro: bool = True):
    network = UserNetwork()
    if os.path.isfile(filepath):
        network.load(filepath)
    counter = 0
    # determine individual leaning
    # using user name instead of id to identify users because banned users don't have an id anymore
    for index, content in data.iterrows():
        counter += 1
        if counter % 1000 == 0:
            print(f"submission number: {counter}")
        # add submission
        network.add_content(content.author_fullname, content.score, pro)
    network.save(filepath)
    print(network)


def analyse(filepath: str):
    print("Analyse...")
    network = UserNetwork()
    if os.path.isfile(filepath):
        network.load(filepath)
    else:
        print("Couldn't load filepath!")
        return
    metrics = network.get_metrics()
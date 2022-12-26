from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            "id": self.id,
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
        df = pd.DataFrame([user.get_metrics() for user in self.users.values() if user.is_relevant()])
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
            "relevant_pro": [user for user in self.pro_users if user.is_relevant()],
            "con": self.con_users,
            "relevant_con": [user for user in self.con_users if user.is_relevant()],
            "undefined": self.undefined_users,
            "relevant_undefined": [user for user in self.undefined_users if user.is_relevant()],
        }

    def __str__(self):
        overview = self.overview()
        return f"Pro users: {len(overview.get('pro'))} (relevant: {len(overview.get('relevant_pro'))})\n" \
               f"Con users: {len(overview.get('con'))} (relevant: {len(overview.get('relevant_con'))})\n" \
               f"Undefined users: {len(overview.get('undefined'))} (relevant: " \
               f"{len(overview.get('relevant_undefined'))})\n" \
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
    print(network)
    metrics = network.get_metrics()
    print(metrics.columns)

    # Analyse amount of posts in relation to votes for both subreddits separately

    posted_in_pro = metrics[metrics["amount_posts_pro"] > 0]
    print(f"Amount of users who posted in pro Clinton: {len(posted_in_pro)}")
    posted_in_pro["group"] = np.select([
        posted_in_pro["amount_posts_pro"].lt(10),
        (posted_in_pro["amount_posts_pro"].ge(10)) & (posted_in_pro["amount_posts_pro"].lt(100)),
        (posted_in_pro["amount_posts_pro"].ge(100)) & (posted_in_pro["amount_posts_pro"].lt(500)),
        posted_in_pro["amount_posts_pro"].ge(500) & posted_in_pro["amount_posts_pro"].lt(5000),
        posted_in_pro["amount_posts_pro"].ge(5000)
    ], [
        "< 10 posts",
        "between 10 and 100 posts",
        "between 100 and 500 posts",
        "between 500 and 5000 posts",
        "more than 5000 posts"
    ])
    posted_in_con = metrics[metrics["amount_posts_con"] > 0]
    print(f"Amount of users who posted in pro Trump: {len(posted_in_con)}")

    posted_in_con["group"] = np.select([
        posted_in_con["amount_posts_con"].lt(10),
        (posted_in_con["amount_posts_con"].ge(10)) & (posted_in_con["amount_posts_con"].lt(100)),
        (posted_in_con["amount_posts_con"].ge(100)) & (posted_in_con["amount_posts_con"]).lt(500),
        posted_in_con["amount_posts_con"].ge(500) & posted_in_con["amount_posts_con"].lt(5000),
        posted_in_con["amount_posts_con"].ge(5000)
    ], [
        "< 10 posts",
        "between 10 and 100 posts",
        "between 100 and 500 posts",
        "between 500 and 5000 posts",
        "more than 5000 posts"
    ])

    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
    axes = axes.ravel()
    axes[0] = sns.histplot(
        ax=axes[0],
        data=posted_in_pro,
        x="amount_posts_pro",
        bins=50
    )
    axes[0].set_title("Amount posts per user on pro Clinton subreddits")
    axes[0].set_yscale("log")
    axes[1] = sns.histplot(
        ax=axes[1],
        data=posted_in_con,
        x="amount_posts_con",
        bins=50
    )
    axes[1].set_title("Amount posts per user on pro Trump subreddits")
    axes[1].set_yscale("log")

    plt.savefig("../plots/amount_posts_distributions_comparison.pdf")

    print("Statistical data for different user groups: Pro Clinton:")
    for group in [
        "< 10 posts",
        "between 10 and 100 posts",
        "between 100 and 500 posts",
        "between 500 and 5000 posts",
        "more than 5000 posts"
    ]:
        group_df = posted_in_pro[posted_in_pro["group"] == group]
        print(f"Group: {group} with {len(group_df)} users:")
        group_mean_score = group_df["average_score_pro"].mean()
        group_05_percentile = group_df["average_score_pro"].quantile(0.05)
        group_25_percentile = group_df["average_score_pro"].quantile(0.25)
        group_75_percentile = group_df["average_score_pro"].quantile(0.75)
        group_95_percentile = group_df["average_score_pro"].quantile(0.95)
        print(f"Mean Score: {group_mean_score}\n0.05 Quantile: {group_05_percentile}\n"
              f"0.25 Quantile: {group_25_percentile}\n0.75 Quantile: {group_75_percentile}\n"
              f"0.95 Quantile: {group_95_percentile}")

    print("Statistical data for different user groups: Pro Trump:")
    for group in [
        "< 10 posts",
        "between 10 and 100 posts",
        "between 100 and 500 posts",
        "between 500 and 5000 posts",
        "more than 5000 posts"
    ]:
        group_df = posted_in_con[posted_in_con["group"] == group]
        print(f"Group: {group} with {len(group_df)} users:")
        group_mean_score = group_df["average_score_con"].mean()
        group_05_percentile = group_df["average_score_con"].quantile(0.05)
        group_25_percentile = group_df["average_score_con"].quantile(0.25)
        group_75_percentile = group_df["average_score_con"].quantile(0.75)
        group_95_percentile = group_df["average_score_con"].quantile(0.95)
        print(f"Mean Score: {group_mean_score}\n0.05 Quantile: {group_05_percentile}\n"
              f"0.25 Quantile: {group_25_percentile}\n0.75 Quantile: {group_75_percentile}\n"
              f"0.95 Quantile: {group_95_percentile}")

    fig, axes = plt.subplots(nrows=2, figsize=(14, 14))
    axes = axes.ravel()
    axes[0] = sns.boxplot(
        data=posted_in_pro,
        x="group",
        y="average_score_pro",
        ax=axes[0],
        order=[
            "< 10 posts",
            "between 10 and 100 posts",
            "between 100 and 500 posts",
            "between 500 and 5000 posts",
            "more than 5000 posts"
        ]
    )
    axes[0].set_title("Relationship between amount of posts and received scores for pro Clinton Subreddits")
    axes[0].set_yscale("log")
    axes[1] = sns.boxplot(
        data=posted_in_con,
        x="group",
        y="average_score_con",
        ax=axes[1],
        order=[
            "< 10 posts",
            "between 10 and 100 posts",
            "between 100 and 500 posts",
            "between 500 and 5000 posts",
            "more than 5000 posts"
        ]
    )
    axes[1].set_title("Relationship between amount of posts and received scores for pro Trump Subreddit")
    axes[1].set_yscale("log")
    plt.savefig("../plots/outsiders-vs-members-comparison.pdf")

    # ----- analyse users who posted in both subreddits

    posted_in_both = metrics[metrics.posted_both]
    only_posted_pro = metrics[metrics["amount_posts_con"] == 0]
    only_posted_con = metrics[metrics["amount_posts_pro"] == 0]
    print(f"Number relevant users: {len(metrics)}")
    print(f"Number of users who posted in both subreddits: {len(posted_in_both)}")

    # plot distribution of who posted who

    metrics = metrics.assign(percentage_pro=metrics["amount_posts_pro"] / (
            metrics["amount_posts_pro"] + metrics["amount_posts_con"]
    )).sort_values("percentage_pro")

    fig, ax = plt.subplots(figsize=(14, 14))
    ax = sns.lineplot(
        data=metrics,
        x="id",
        y="percentage_pro",
        ax=ax,
    )
    plt.xticks([])
    plt.savefig("../plots/posted_in_both_percentages.pdf")

    # calculate percentages of downvoted posts for different user groups

    dfs_downvoted_percentages = {
        "only_pro": only_posted_pro.assign(
            percentage_downvoted=only_posted_pro["amount_downvoted_pro"] / only_posted_pro["amount_posts_pro"]
        ).assign(group="Only pro Clinton"),
        "only_con": only_posted_con.assign(
            percentage_downvoted=only_posted_con["amount_downvoted_con"] / only_posted_con["amount_posts_con"]
        ).assign(group="Only pro Trump"),
        "both_pro": posted_in_both.assign(
            percentage_downvoted=posted_in_both["amount_downvoted_pro"] / posted_in_both["amount_posts_pro"]
        ).assign(group="Posted in both; downvotes in pro Clinton"),
        "both_con": posted_in_both.assign(
            percentage_downvoted=posted_in_both["amount_downvoted_con"] / posted_in_both["amount_posts_con"]
        ).assign(group="Posted in both; downvotes in pro Trump")
    }
    mean_downvoted_percentages = {
        "only_pro": dfs_downvoted_percentages["only_pro"]["percentage_downvoted"].mean(),
        "only_con": dfs_downvoted_percentages["only_con"]["percentage_downvoted"].mean(),
        "both_pro": dfs_downvoted_percentages["both_pro"]["percentage_downvoted"].mean(),
        "both_con": dfs_downvoted_percentages["both_con"]["percentage_downvoted"].mean(),
    }
    percentages_of_downvoted_users = {
        index: len(dfs_downvoted_percentages[index][
                       dfs_downvoted_percentages[index]["percentage_downvoted"] == 0
                       ]) / len(dfs_downvoted_percentages[index]) for index in
        ["only_pro", "only_con", "both_pro", "both_con"]
    }
    print(f"Mean of downvoted posts percentages: {mean_downvoted_percentages}")
    print(f"Percentage of users where no content was downvoted: {percentages_of_downvoted_users}")

    # plot the distribution of how many downvoted posts users made

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14, 14))
    axes = axes.ravel()

    sns.histplot(
        data=dfs_downvoted_percentages["only_pro"],
        ax=axes[0],
        x="percentage_downvoted",
        bins=30
    )
    axes[0].set_title("Only posted in pro Clinton subreddit")
    sns.histplot(
        data=dfs_downvoted_percentages["only_con"],
        ax=axes[1],
        x="percentage_downvoted",
        bins=30,
    )
    axes[1].set_title("Only posted in pro Trump subreddit")
    sns.histplot(
        data=dfs_downvoted_percentages["both_pro"],
        ax=axes[2],
        x="percentage_downvoted",
        bins=30,
    )
    axes[2].set_title("Posted in both; % of downvoted content in pro Clinton subreddit")
    sns.histplot(
        data=dfs_downvoted_percentages["both_con"],
        ax=axes[3],
        x="percentage_downvoted",
        bins=30,
    )
    axes[3].set_title("Posted in both; % of downvoted content in pro Trump subreddit")
    plt.tight_layout()
    plt.savefig("../plots/downvoted_dists.pdf")

    # use a violinplot to gain further insights on the same distributin

    df_merged_for_plot = pd.concat([
        df[df["percentage_downvoted"] > 0][[
            "group",
            "percentage_downvoted"
        ]] for df in dfs_downvoted_percentages.values()
    ])

    fig, ax = plt.subplots(figsize=(16, 14))
    ax = sns.violinplot(
        data=df_merged_for_plot,
        x="group",
        y="percentage_downvoted",
        split=True,
        showmedians=True,
        showmeans=True,
        bw=0.1,
        width=1
    )
    ax.set(xlabel=None)
    plt.savefig("../plots/downvoted_nonnull_dists_violin.pdf")

    # 2d displot showing the amount of posts on pro Clinton/Trump subreddits vs the amount of downvoted posts

    dfs_for_2d_displot = {
        "both_pro": dfs_downvoted_percentages["both_pro"].assign(
            percentage_posts_in_pro_clinton=dfs_downvoted_percentages["both_pro"]["amount_posts_pro"] / (
                    dfs_downvoted_percentages["both_pro"]["amount_posts_pro"] +
                    dfs_downvoted_percentages["both_pro"]["amount_posts_con"]
            )
        ),
        "both_con": dfs_downvoted_percentages["both_con"].assign(
            percentage_posts_in_pro_clinton=dfs_downvoted_percentages["both_con"]["amount_posts_pro"] / (
                    dfs_downvoted_percentages["both_con"]["amount_posts_pro"] +
                    dfs_downvoted_percentages["both_con"]["amount_posts_con"]
            )
        ),
    }

    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
    axes = axes.ravel()

    sns.histplot(
        data=dfs_for_2d_displot["both_pro"],
        x="percentage_downvoted",
        y="percentage_posts_in_pro_clinton",
        ax=axes[0],
        bins=25,
        binrange=((0, 1), (0, 1)),
        cbar_kws=dict(spacing="proportional"),
        thresh=5,
    )
    axes[0].set_title("Percentage of downvoted posts in pro Clinton subreddit")
    # related to whether posts were posted in pro Clinton or pro Trump subreddit
    sns.histplot(
        data=dfs_for_2d_displot["both_con"],
        x="percentage_downvoted",
        y="percentage_posts_in_pro_clinton",
        ax=axes[1],
        bins=25,
        binrange=((0, 1), (0, 1)),
        thresh=5,
        cbar=True
    )
    axes[1].set_title("Percentage of downvoted posts in pro Trump subreddit")
    plt.tight_layout()
    plt.savefig("../plots/downvotes_vs_post_dist.pdf")

    # same plot, filtered by users without any downvoted posts

    dfs_for_2d_displot_filtered = {
        key: df[df["percentage_downvoted"] > 0] for key, df in dfs_for_2d_displot.items()
    }

    fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
    axes = axes.ravel()
    sns.histplot(
        data=dfs_for_2d_displot_filtered["both_pro"],
        x="percentage_downvoted",
        y="percentage_posts_in_pro_clinton",
        ax=axes[0],
        bins=25,
        binrange=((0, 1), (0, 1)),
        cbar_kws=dict(spacing="proportional"),
        thresh=5,
    )
    axes[0].set_title("Percentage of downvoted posts in pro Clinton subreddit")
    # related to whether posts were posted in pro Clinton or pro Trump subreddit
    sns.histplot(
        data=dfs_for_2d_displot_filtered["both_con"],
        x="percentage_downvoted",
        y="percentage_posts_in_pro_clinton",
        ax=axes[1],
        bins=25,
        binrange=((0, 1), (0, 1)),
        thresh=5,
        cbar=True
    )
    axes[1].set_title("Percentage of downvoted posts in pro Trump subreddit")
    plt.tight_layout()
    plt.savefig("../plots/downvotes_vs_post_dist_filtered_no_downvotes.pdf")




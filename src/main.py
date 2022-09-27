from httpClient import HttpClient
from analysis import build_network_leaning, analyse
import datetime
import calendar


def get_data(start_month: int, start_year: int, end_month: int, end_year: int, subreddit: str):
    http_client = HttpClient()
    data = http_client.fetch_submissions(
        subreddit,
        datetime.datetime(
           start_year, start_month, 1, 0, 0
        ),
        datetime.datetime(
            end_year, end_month, calendar.monthrange(end_year, end_month)[1], 0, 0
        )
    )
    return data


def gather_data():
    dataset = get_data(
        1,
        2016,
        12,
        2016,
        "HillaryForAmerica"
    )
    print(f"amount submissions: {len(dataset)}")
    build_network_leaning(dataset, "../saves/test2.json", pro=True)
    # PRO = Clinton, CON = Trump


def analysis():
    analyse("../saves/test2.json")


analysis()

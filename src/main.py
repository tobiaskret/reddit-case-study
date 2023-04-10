from httpClient import HttpClient
from analysis import build_network_leaning, analyse
import sys
import datetime
import calendar


RUN_MODES = {
    "GATHER_DATA": 0,
    "ANALYSE": 1
}


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


def gather_data(save_path, subreddit, pro=True):
    dataset = get_data(
        1,
        2016,
        12,
        2016,
        subreddit
    )
    print(f"amount submissions: {len(dataset)}")
    build_network_leaning(dataset, save_path, pro)
    # PRO = Clinton, CON = Trump


def analysis(save_path):
    analyse(save_path)


args = sys.argv[1:]  # First parameter is always filename "main.py"
# default values if no parameters are given
mode = RUN_MODES["ANALYSE"]
filepath = "../saves/save.json"
# cl parameters: first is run mode, second is path to save file to.
if len(args) > 0:
    if args[0] in RUN_MODES:
        mode = RUN_MODES[args[0]]
    if args[0] in RUN_MODES.values():
        mode = args[0]
    if len(args) > 1:
        filepath = args[1]
# run
if mode:
    analysis(filepath)
else:
    # currently for the timeframe 2016. If other time frame is desired, make the date values in gather_data parameters
    gather_data(filepath, "The_Donald", False)
    gather_data(filepath, "hillaryclinton")
    gather_data(filepath, "HillaryForAmerica")


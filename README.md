## Case study: pro Trump and pro Clinton communities on Reddit

This repository contains code used in my study project _Using users' votes to detect echo chambers in social media_ during my masters program in Cognitive Science at Universität Osnabrück.

#### How to run

Run the src/main.py with up to 2 command line parameters. The first one sets the run mode: either "GATHER_DATA" ("0" does the same) or "ANALYSE" ("1"). The second parameters sets the filepath where to save the data (in mode 0) or take the data from (in mode 1). By default, the script collects the data of pro Trump and pro Clinton communities on Reddit for the year 2016. To change this behaviour, adjust the main.py file.

Gathering data requires an internet connection. Gathering data takes a long time (due to HTTP request limits of the Pushshift dataset), consider splitting it into multiple steps by adjusting the code.

Plots are automatically saved to the plots directory. To create different plots, change the src/analysis.py

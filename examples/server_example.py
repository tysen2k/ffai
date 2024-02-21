#!/usr/bin/env python3
import os
from scripted_bot_example import *
from grodbot import *
from riskbot import *

import botbowl.web.server as server

if __name__ == "__main__":
    if os.path.exists("/.dockerenv"):
        # If you're running in a docker container, you want the server to be accessible from outside the container.
        host = "0.0.0.0"
    else:
        # If you're running locally, you want the server to be accessible from localhost only b/c of security.
        host = "127.0.0.1"

from botbowl.ai.registry import make_bot
botbowl.web.api.new_game(home_team_name="Human Team",
             away_team_name="Human Team",
             home_agent=make_bot("riskbot"),
             away_agent=make_bot("riskbot"),
             game_mode='standard')

server.start_server(host=host, debug=True, use_reloader=False, port=1234)

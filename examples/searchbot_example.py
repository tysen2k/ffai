#!/usr/bin/env python3

from ffai.core import Agent, Game
from ffai.core.load import *
from ffai.ai.registry import register_bot, make_bot

import examples.scripted_bot_example

# Load configurations, rules, arena and teams
config = load_config("bot-bowl-i.json")
ruleset = load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
arena = load_arena(config.arena)
home = load_team_by_filename("human", ruleset)
away = load_team_by_filename("human", ruleset)
config.competition_mode = False

# Play 5 games as away
for i in range(5):
    away_agent = make_bot('searchbot')
    away_agent.name = 'searchbot'
    home_agent = make_bot('searchbot')
    home_agent.name = 'searchbot'
    config.debug_mode = False
    game = Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    print("Starting game", (i + 1))
    start = time.time()
    game.init()
    end = time.time()
    print(end - start)

# Play 5 games as home
for i in range(5):
    away_agent = make_bot('searchbot')
    away_agent.name = 'searchbot'
    home_agent = make_bot('searchbot')
    home_agent.name = 'searchbot'
    config.debug_mode = False
    game = Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    print("Starting game", (i + 1))
    start = time.time()
    game.init()
    end = time.time()
    print(end - start)


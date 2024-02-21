#!/usr/bin/env python3
import botbowl
import botbowl.core.model as m
import botbowl.core.table as t
import botbowl.core.procedure as p
import botbowl.core.pathfinding as pf
from botbowl import Action, ActionType, Square, BBDieResult, Skill, Formation, ProcBot
from typing import Optional, List, Dict, Tuple
import botbowl.core.game as g
import numpy as np

"""
GrodBot
Author: Peter Moore

Quick Technical Description:
The entry point for the GrodBot from the FFAI client is the "act" function.  In this, the "turn" function is probably
the most important.  This function is where the Bot selects which player to take an action with, and which squares to
move to, block, foul etc.  When the "act" function is called, when the "Turn" proc is next on the stack, no player is
currently selected.  When a player is mid-turn, then a PlayerAction proc is next on the stack instead.  The GrodBot
deals with this by setting an internal variable "current_move" when the turn function is called, which contains the
entire sequence of desired moves for a sinple player.  When the PlayerAction proc is on the stack, the player_action
function is called which will simply try to return the next Action in the current_move variable.

A scripted bot that iteratively evaluates all possible actions, assigning a score for each, and executes the highest performing score next. 
Patterns of behaviour develop from the combination of individually simple rules such that the bot will attempt to blitz ball carriers with
assistance, move players to open receiving positions and pass to them, "cage" the ball carrier to protect it etc. Improvements and bug fixes have been applied since last years 
entry, particularly around less use of screens, more emphasis on chasing the ball and a large number of bug fixes to the desired behaviour correct.


For the turn function, the approach is to calculate all possible lowest probability moves of all players 
(using path finding) and assign a "score" to each move.  The highest score move of all the possibilities is the chosen
action and the one that is set to the current_move.
"""


class GrodBot(ProcBot):
    """
    A Bot that uses path finding to evaluate all possibilities.

    WIP!!! Hand-offs and Pass actions going a bit funny.

    """

    mean_actions_available = []
    steps = []

    BASE_SCORE_BLITZ = 60.0
    BASE_SCORE_FOUL = -50.0
    BASE_SCORE_BLOCK = 65   # For a two dice block
    BASE_SCORE_HANDOFF = 40.0
    BASE_SCORE_PASS = 40.0
    BASE_SCORE_MOVE_TO_OPPONENT = 45.0
    BASE_SCORE_MOVE_BALL = 65.0
    BASE_SCORE_MOVE_TOWARD_BALL = 40.0
    BASE_SCORE_MOVE_TO_SWEEP = 0.0
    BASE_SCORE_CAGE_BALL = 70.0
    BASE_SCORE_MOVE_TO_BALL = 145.0
    BASE_SCORE_BALL_AND_CHAIN = 75.0
    BASE_SCORE_DEFENSIVE_SCREEN = -15.0
    ADDITIONAL_SCORE_DODGE = 100.0  # Lower this value to dodge more.
    ADDITIONAL_SCORE_NEAR_SIDELINE = -20.0
    ADDITIONAL_SCORE_SIDELINE = -40.0
    ADDITIONAL_SCORE_PRONE = 15.0  # Favour moving prone players earlier

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.opp_team = None
        self.current_move: Optional[ActionSequence] = None
        self.verbose = True
        self.debug = False
        self.heat_map: Optional[FfHeatMap] = None
        self.actions_available = []

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_debug(self, debug):
        self.debug = debug

    def act(self, game):

        # Refresh my_team and opp_team (they seem to be copies)
        proc = game.get_procedure()
        available_actions = game.state.available_actions
        available_action_types = [available_action.action_type for available_action in available_actions]

        # Update local my_team and opp_team variables to latest copy (to ensure fresh data)
        if hasattr(proc, 'team'):
            assert proc.team == self.my_team
            self.my_team = proc.team
            self.opp_team = game.get_opp_team(self.my_team)

        # For statistical purposes, keeps a record of # action choices.
        available = 0
        for action_choice in available_actions:
            if len(action_choice.positions) == 0 and len(action_choice.players) == 0:
                available += 1
            elif len(action_choice.positions) > 0:
                available += len(action_choice.positions)
            else:
                available += len(action_choice.players)
        self.actions_available.append(available)

        # Evaluate appropriate action for each possible procedure
        if isinstance(proc, p.CoinTossFlip):
            action = self.coin_toss_flip(game)
        elif isinstance(proc, p.CoinTossKickReceive):
            action = self.coin_toss_kick_receive(game)
        elif isinstance(proc, p.Setup):
            action = self.setup(game)
        elif isinstance(proc, p.PlaceBall):
            action = self.place_ball(game)
        elif isinstance(proc, p.HighKick):
            action = self.high_kick(game)
        elif isinstance(proc, p.Touchback):
            action = self.touchback(game)
        elif isinstance(proc, p.Turn):
            if proc.quick_snap:
                action = self.quick_snap(game)
            elif proc.blitz:
                action = self.blitz(game)
            else:
                action = self.turn(game)
        elif isinstance(proc, p.MoveAction):
            action = self.player_action(game)
        elif isinstance(proc, p.Block):
            action = self.block(game)
        elif isinstance(proc, p.BlockAction):
            action = self.player_action(game)
        elif isinstance(proc, p.Push):
            action = self.push(game)
        elif isinstance(proc, p.FollowUp):
            action = self.follow_up(game)
        elif isinstance(proc, p.Apothecary):
            action = self.apothecary(game)
        # elif isinstance(proc, p.PassAction):
        #     action = self.pass_action(game)
        elif isinstance(proc, p.Interception):
            action = self.interception(game)
        elif isinstance(proc, p.Reroll):
            action = self.reroll(game)
        elif isinstance(proc, p.Shadowing):
            action = self.shadowing(game)
        else:
            if self.debug:
                raise Exception("Unknown procedure: ", proc)
            elif t.ActionType.USE_SKILL in available_action_types:
                # Catch-all for things like Break Tackle, Diving Tackle etc
                return m.Action(t.ActionType.USE_SKILL)
            else:
                # Ugly catch-all -> simply pick an action
                action_choice = available_actions[0]
                player = action_choice.players[0] if action_choice.players else None
                position = action_choice.positions[0] if action_choice.positions else None
                action = m.Action(action_choice.action_type, position=position, player=player)
                # raise Exception("Unknown procedure: ", proc)

        # Check returned Action is valid
        action_found = False
        for available_action in available_actions:
            if isinstance(action.action_type, type(available_action.action_type)):
                if available_action.players and available_action.positions:
                    action_found = (action.player in available_action.players) and (action.player in available_action.players)
                elif available_action.players:
                    action_found = action.player in available_action.players
                elif available_action.positions:
                    action_found = action.position in available_action.positions
                else:
                    action_found = True
        if not action_found:
            if self.debug:
                raise Exception('Invalid action')
            else:
                # Ugly catch-all -> simply pick an action
                action_choice = available_actions[0]
                player = action_choice.players[0] if action_choice.players else None
                position = action_choice.positions[0] if action_choice.positions else None
                action = m.Action(action_choice.action_type, position=position, player=player)

        # if self.verbose:
        #     current_team = game.state.current_team.name if game.state.current_team is not None else available_actions[0].team.name
        #     print('      Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + current_team + ', Action=' + action.action_type.name)

        return action

    def reroll(self, game):
        proc = game.get_procedure()
        # target_roll = proc.context.roll.target
        # target_higher = proc.context.roll.target_higher
        # dice = proc.context.roll.dice
        # num_dice = len(dice)
        if proc.can_use_pro:
            return m.Action(t.ActionType.USE_SKILL)
        if isinstance(proc.context, p.GFI):
            return m.Action(t.ActionType.USE_REROLL)
        if isinstance(proc.context, p.Dodge):
            return m.Action(t.ActionType.USE_REROLL)
        if isinstance(proc.context, p.Catch):
            return m.Action(t.ActionType.USE_REROLL)
        if isinstance(proc.context, p.Pickup):
            return m.Action(t.ActionType.USE_REROLL)
        else:
            return m.Action(t.ActionType.USE_REROLL)

    def new_game(self, game: g.Game, team):
        """
        Called when a new game starts.
        """
        self.my_team = team
        self.opp_team = game.get_opp_team(team)
        self.actions_available = []

    def coin_toss_flip(self, game: g.Game):
        """
        Select heads/tails and/or kick/receive
        """
        return m.Action(t.ActionType.TAILS)
        # return Action(ActionType.HEADS)

    def coin_toss_kick_receive(self, game: g.Game):
        """
        Select heads/tails and/or kick/receive
        """
        return m.Action(t.ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game: g.Game) -> m.Action:
        """
        Move players from the reserves to the pitch
        """

        if isinstance(game.get_procedure(), p.Setup):
            proc: p.Setup = game.get_procedure()
        else:
            raise ValueError('Setup procedure expected')

        if proc.reorganize:
            # We are dealing with perfect defence.  For now do nothing, but we could send all players back to reserve box
            action_steps: List[m.Action] = [m.Action(t.ActionType.END_SETUP)]
            self.current_move = ActionSequence(action_steps, description='Perfect Defence do nothing')

        else:

            if not BotHelper.get_players(game, self.my_team, include_own=True, include_opp=False, include_off_pitch=False):
                # If no players are on the pitch yet, create a new ActionSequence for the setup.
                action_steps: List[m.Action] = []

                turn = game.state.round
                half = game.state.half
                opp_score = 0
                for team in game.state.teams:
                    if team != self.my_team:
                        opp_score = max(opp_score, team.state.score)
                score_diff = self.my_team.state.score - opp_score

                # Choose 11 best players to field
                players_available: List[m.Player] = []
                for available_action in game.state.available_actions:
                    if available_action.action_type == t.ActionType.PLACE_PLAYER:
                        players_available = available_action.players

                players_sorted_value = sorted(players_available, key=lambda x: BotHelper.player_value(game, x), reverse=True)
                n_keep: int = min(11, len(players_sorted_value))
                players_available = players_sorted_value[:n_keep]

                players_sorted_bash = sorted(players_available, key=lambda x: BotHelper.player_bash_ability(game, x), reverse=True)
                players_sorted_blitz = sorted(players_available, key=lambda x: BotHelper.player_blitz_ability(game, x), reverse=True)
                players_sorted_value = sorted(players_available, key=lambda x: BotHelper.player_value(game, x), reverse=False)
                players_sorted_pass = sorted(players_available, key=lambda x: BotHelper.player_pass_ability(game, x), reverse=True)
                players_added = []

                if game.get_receiving_team() == self.my_team:
                    # Receiving
                    place_squares: List[m.Square] = [
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 7),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 8),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 9),
                        # Receiver next
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 7), 8),
                        # Support line players
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 10),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 6),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 12), 4),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 12), 12),
                        # A bit wide semi-defensive - want catchers here
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 11), 3),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 11), 13),
                        # Extra help at the back
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 10), 8)
                    ]

                    for i in range(min(11, len(players_available))):
                        place_square = place_squares.pop(0)

                        found = False
                        if i in [3, 10]:
                            # 4th player and 11 player are receiving type players
                            while not found:
                                player = players_sorted_pass.pop(0)
                                found = player not in players_added
                        elif i in [0, 1, 2, 4, 5]:
                            # These are my "bash" players"
                            while not found:
                                player = players_sorted_bash.pop(0)
                                found = player not in players_added
                        else:
                            # Everyone else
                            while not found:
                                player = players_sorted_blitz.pop(0)
                                found = player not in players_added

                        players_added.append(player)
                        action_steps.append(m.Action(t.ActionType.PLACE_PLAYER, player=player, position=place_square))
                else:
                    # Kicking
                    place_squares: List[m.Square] = [
                        # LOS squares first
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 5),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 6),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 13), 7),

                        # in close support next
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 12), 8),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 12), 10),

                        # wings
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 12), 3),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 12), 13),


                        # in close support second row
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 11), 9),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 11), 11),

                        # wings second row
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 11), 2),
                        game.get_square(BotHelper.reverse_x_for_right(game, self.my_team, 11), 14)
                        ]

                    for i in range(min(11, len(players_available))):
                        place_square = place_squares.pop(0)
                        found = False
                        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            # 4th player and 11 player are receiving type players
                            while not found:
                                player = players_sorted_bash.pop(0)
                                found = player not in players_added

                        players_added.append(player)
                        action_steps.append(m.Action(t.ActionType.PLACE_PLAYER, player=player, position=place_square))

                action_steps.append(m.Action(t.ActionType.END_SETUP))
                self.current_move = ActionSequence(action_steps, description='Setup')

        # We must have initialised the action sequence, lets execute it
        if self.current_move.is_empty():
            raise Exception('what')
        else:
            next_action: m.Action = self.current_move.popleft()
        return next_action

    def place_ball(self, game: g.Game):
        """
        Place the ball when kicking.
        """

        # Note left_center square is 7,8
        center_opposite: m.Square = m.Square(BotHelper.reverse_x_for_left(game, self.my_team, 7), 8)
        return m.Action(t.ActionType.PLACE_BALL, position=center_opposite)

    def high_kick(self, game: g.Game):
        """
        Select player to move under the ball.
        """
        # Get the list of players available to place
        players_available: List[m.Player] = []
        for available_action in game.state.available_actions:
            if available_action.action_type == t.ActionType.PLACE_PLAYER:
                players_available = available_action.players
        ball_pos = game.get_ball_position()

        if game.is_team_side(game.get_ball_position(), self.my_team) and game.get_player_at(game.get_ball_position()) is None:
            if players_available:
                players_sorted = sorted(players_available, key=lambda x: BotHelper.player_blitz_ability(game, x), reverse=True)
                player = players_sorted[0]
                return m.Action(t.ActionType.SELECT_PLAYER, player=player)
        return m.Action(t.ActionType.SELECT_NONE)

    def touchback(self, game: g.Game):
        """
        Select player to give the ball to.
        """
        players_available = game.get_players_on_pitch(self.my_team, up=True)
        if players_available:
            players_sorted = sorted(players_available, key=lambda x: BotHelper.player_blitz_ability(game, x), reverse=True)
            player = players_sorted[0]
            return m.Action(t.ActionType.SELECT_PLAYER, player=player)
        return m.Action(t.ActionType.SELECT_NONE)

    def set_next_move(self, game: g.Game):
        """ Set self.current_move

        :param game:
        """
        self.current_move = None

        players_moved: List[m.Player] = BotHelper.get_players(game, self.my_team, include_own=True, include_opp=False, include_used=True, only_used=False)
        players_to_move: List[m.Player] = BotHelper.get_players(game, self.my_team, include_own=True, include_opp=False, include_used=False)
        paths_own: Dict[m.Player, List[pf.Path]] = dict()
        for player in players_to_move:
            paths = pf.get_all_paths(game, player, from_position=None)
            paths_own[player] = paths

        players_opponent: List[m.Player] = BotHelper.get_players(game, self.my_team, include_own=False, include_opp=True, include_stunned=False)
        paths_opposition: Dict[m.Player, List[pf.Path]] = dict()
        for player in players_opponent:
            paths = pf.get_all_paths(game, player, from_position=None)
            paths_opposition[player] = paths

        # Create a heat-map of control zones
        heat_map: FfHeatMap = FfHeatMap(game, self.my_team)
        heat_map.add_unit_by_paths(game, paths_opposition)
        heat_map.add_unit_by_paths(game, paths_own)
        heat_map.add_players_moved(game, BotHelper.get_players(game, self.my_team, include_own=True, include_opp=False, only_used=True))
        self.heat_map = heat_map

        all_actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            if action_choice.action_type == t.ActionType.START_MOVE:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = paths_own[player]
                    all_actions.extend(BotHelper.potential_move_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.START_BLITZ:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = pf.get_all_paths(game, player, blitz=True)
                    for path in paths:
                        if game.get_player_at(path.get_last_step()) is None:
                            continue
                        all_actions.extend(BotHelper.potential_blitz_actions(game, heat_map, player, path))
            elif action_choice.action_type == t.ActionType.START_FOUL:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = paths_own[player]
                    all_actions.extend(BotHelper.potential_foul_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.START_BLOCK:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    all_actions.extend(BotHelper.potential_block_actions(game, heat_map, player))
            elif action_choice.action_type == t.ActionType.START_PASS:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.get_ball_position() == player_square:
                        paths = paths_own[player]
                        all_actions.extend(BotHelper.potential_pass_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.START_HANDOFF:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.get_ball_position() == player_square:
                        paths = paths_own[player]
                        all_actions.extend(BotHelper.potential_handoff_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.END_TURN:
                all_actions.extend(BotHelper.potential_end_turn_action(game))

        if all_actions:
            all_actions.sort(key=lambda x: x.score, reverse=True)

            f = [x.score for x in all_actions]

            found = False
            while not found:
                best_move = all_actions.pop(0)
                if BotHelper.is_do_nothing(best_move):
                    print('Best move - do nothing.  Remove')
                    # Best move is for a particular player to do nothing. Remove all actions from the list that correspond to that player. We may decide to move that player later anyway.
                    all_actions = [action for action in all_actions if action.player != best_move.player]
                else:
                    found = True

            self.current_move = best_move
            if self.verbose:
                print('   Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + game.state.current_team.name + ', Action=' + self.current_move.description + ', Score=' + str(self.current_move.score))

    def set_continuation_move(self, game: g.Game):
        """ Set self.current_move

        :param game:
        """
        self.current_move = None

        player: m.Player = game.state.active_player
        paths = pf.get_all_paths(game, player, blitz=False)

        all_actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            if action_choice.action_type == t.ActionType.MOVE:
                players_available: List[m.Player] = action_choice.players
                all_actions.extend(BotHelper.potential_move_actions(game, self.heat_map, player, paths, is_continuation=True))
            elif action_choice.action_type == t.ActionType.END_PLAYER_TURN:
                all_actions.extend(BotHelper.potential_end_player_turn_action(game, self.heat_map, player))

        if all_actions:
            all_actions.sort(key=lambda x: x.score, reverse=True)

            f = [x.score for x in all_actions]

            self.current_move = all_actions[0]

            if self.verbose:
                print('   Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + game.state.current_team.name + ', Action=Continue Move + ' + self.current_move.description + ', Score=' + str(self.current_move.score))

    def turn(self, game: g.Game) -> m.Action:
        """
        Start a new player action / turn.
        """

        # Simple algorithm:
        #   Loop through all available (yet to move) players.
        #   Compute all possible moves for all players.
        #   Assign a score to each action for each player.
        #   The player/play with the highest score is the one the Bot will attempt to use.
        #   Store a representation of this turn internally (for use by player-action) and return the action to begin.

        self.set_next_move(game)
        next_action: m.Action = self.current_move.popleft()
        return next_action

    def quick_snap(self, game: g.Game):

        self.current_move = None
        return m.Action(t.ActionType.END_TURN)

    def blitz(self, game: g.Game):

        return self.turn(game)

    def player_action(self, game: g.Game):
        """
        Take the next action from the current stack and execute
        """
        if self.current_move.is_empty():
            self.set_continuation_move(game)

        action_step = self.current_move.popleft()
        return action_step

    def shadowing(self, game: g.Game):
        """
        Select block die or reroll.
        """
        # Loop through available dice results
        proc = game.get_procedure()
        return m.Action(t.ActionType.USE_SKILL)

    def block(self, game: g.Game):
        """
        Select block die or reroll.
        """
        # Loop through available dice results
        proc = game.get_procedure()
        if proc.waiting_juggernaut:
            return m.Action(t.ActionType.USE_SKILL)
        if proc.waiting_wrestle_attacker or proc.waiting_wrestle_defender:
            return m.Action(t.ActionType.USE_SKILL)

        active_player: m.Player = game.state.active_player
        attacker: m.Player = game.state.stack.items[-1].attacker
        defender: m.Player = game.state.stack.items[-1].defender
        favor: m.Team = game.state.stack.items[-1].favor

        actions: List[ActionSequence] = []
        check_reroll = False
        for action_choice in game.state.available_actions:
            if action_choice.action_type == t.ActionType.USE_REROLL:
                check_reroll = True
                continue
            action_steps: List[m.Action] = [
                m.Action(action_choice.action_type)
                ]
            score = BotHelper.block_favourability(action_choice.action_type, self.my_team, active_player, attacker, defender, favor)
            actions.append(ActionSequence(action_steps, score=score, description='Block die choice'))

        if check_reroll and BotHelper.check_reroll_block(game, self.my_team, actions, favor):
            return m.Action(t.ActionType.USE_REROLL)
        else:
            actions.sort(key=lambda x: x.score, reverse=True)
            current_move = actions[0]
            return current_move.action_steps[0]

    def push(self, game: g.Game):
        """
        Select square to push to.
        """
        # Loop through available squares
        block_proc: Optional[p.Block] = BotHelper.last_block_proc(game)
        attacker: m.Player = block_proc.attacker
        defender: m.Player = block_proc.defender
        is_blitz_action = block_proc.blitz
        score: float = -100.0
        for to_square in game.state.available_actions[0].positions:
            cur_score = BotHelper.score_push(game, defender.position, to_square)
            if cur_score > score:
                score = cur_score
                push_square = to_square
        return m.Action(t.ActionType.PUSH, position=push_square)

    def follow_up(self, game: g.Game):
        """
        Follow up or not. ActionType.FOLLOW_UP must be used together with a position.
        """
        player = game.state.active_player
        do_follow = BotHelper.check_follow_up(game)
        for position in game.state.available_actions[0].positions:
            if do_follow and player.position != position:
                return m.Action(t.ActionType.FOLLOW_UP, position=position)
            elif not do_follow and player.position == position:
                return m.Action(t.ActionType.FOLLOW_UP, position=position)

    def apothecary(self, game: g.Game):
        """
        Use apothecary?
        """
        # Update here -> apothecary BH in first half, KO or BH in second half
        return m.Action(t.ActionType.USE_APOTHECARY)
        # return Action(ActionType.DONT_USE_APOTHECARY)

    def interception(self, game: g.Game):
        """
        Select interceptor.
        """
        for action in game.state.available_actions:
            if action.action_type == t.ActionType.SELECT_PLAYER:
                for player, rolls in zip(action.players, action.rolls):
                    return m.Action(t.ActionType.SELECT_PLAYER, player=player)
        return m.Action(t.ActionType.SELECT_NONE)

    def pass_action(self, game: g.Game):
        """
        Reroll or not.
        """
        return m.Action(t.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def end_game(self, game: g.Game):
        """
        Called when a game end.
        """
        print(f'''Result for {self.name}''')
        print('------------------')
        print(f'''Num steps: {len(self.actions_available)}''')
        print(f'''Avg. branching factor: {np.mean(self.actions_available)}''')
        GrodBot.steps.append(len(self.actions_available))
        GrodBot.mean_actions_available.append(np.mean(self.actions_available))
        print(f'''Avg. Num steps: {np.mean(GrodBot.steps)}''')
        print(f'''Avg. overall branching factor: {np.mean(GrodBot.mean_actions_available)}''')
        winner = game.get_winner()
        print(f'''Casualties: {game.state.home_team.name} ({game.home_agent.name}): {game.num_casualties(game.state.home_team)} ... {game.state.away_team.name}  ({game.away_agent.name}): {game.num_casualties(game.state.away_team)}''')
        print(f'''Score: {game.state.home_team.name} ({game.home_agent.name}): {game.state.home_team.state.score} ... {game.state.away_team.name}  ({game.away_agent.name}): {game.state.away_team.state.score}''')
        if winner is None:
            print(f'''It's a draw''')
        elif winner == self:
            print(f'''I won''')
        else:
            print(f'''I lost''')
        print('------------------')


class ActionSequence:

    def __init__(self, action_steps: List[m.Action], player: m.Player = None, score: float = 0, description: str = ''):
        """ Creates a new ActionSequence - an ordered list of sequential Actions to attempt to undertake.
        :param action_steps: Sequence of action steps that form this action.
        :param score: A score representing the attractiveness of the move (default: 0)
        :param description: A debug string (default: '')
        """

        # Note the intention of this object is that when the object is acting, as steps are completed,
        # they are removed from the move_sequence so the next move is always the top of the move_sequence
        # lis

        self.action_steps = action_steps
        self.score = score
        self.description = description
        self.player = player

    def is_valid(self, game: g.Game) -> bool:
        pass

    def popleft(self):
        return self.action_steps.pop(0)
        # val = self.action_steps[0]
        # del self.action_steps[0]
        # return val

    def is_empty(self):
        return not self.action_steps


class FfHeatMap:
    """ A heat map of a Blood Bowl field.

    A class for analysing zones of control for both teams
    """

    def __init__(self, game: g.Game, team: m.Team):
        self.game = game
        self.team = team
        # Note that the edges are not on the field, but represent crowd squares
        self.units_friendly: List[List[float]] = [[0.0 for y in range(game.state.pitch.height)] for x in range(game.state.pitch.width)]
        self.units_opponent: List[List[float]] = [[0.0 for y in range(game.state.pitch.height)] for x in range(game.state.pitch.width)]

    def add_unit_paths(self, player: m.Player, paths: List[pf.Path]):
        is_friendly: bool = player.team == self.team

        for path in paths:
            if is_friendly:
                self.units_friendly[path.get_last_step().x][path.get_last_step().y] += path.prob * path.prob
            else:
                self.units_opponent[path.get_last_step().x][path.get_last_step().y] += path.prob * path.prob

    def add_unit_by_paths(self, game: g.Game, paths: Dict[m.Player, List[pf.Path]]):
        for player in paths.keys():
            self.add_unit_paths(player, paths[player])

    def add_players_moved(self, game: g.Game, players: List[m.Player]):
        for player in players:
            adjacents: List[m.Square] = game.get_adjacent_squares(player.position, occupied=True)
            self.units_friendly[player.position.x][player.position.y] += 1.0
            for adjacent in adjacents:
                self.units_friendly[player.position.x][player.position.y] += 0.5

    def get_ball_move_square_safety_score(self, square: m.Square) -> float:

        # Basic idea - identify safe regions to move the ball towards
        # friendly_heat: float = self.units_friendly[square.x][square.y]
        opponent_heat: float = self.units_opponent[square.x][square.y]

        score: float = 30.0 * max(0.0, (1.0 - opponent_heat / 2))

        # score: float=0.0
        # if opponent_heat < 0.25: score += 15.0
        # if opponent_heat < 0.05: score += 15.0
        # if opponent_heat < 1.5: score += 5
        # if friendly_heat > 3.5: score += 10.0
        # score += max(30.0, 5.0*(friendly_heat-opponent_heat))

        return score

    def get_cage_necessity_score(self, square: m.Square) -> float:
        # opponent_friendly: float = self.units_friendly[square.x][square.y]
        opponent_heat: float = self.units_opponent[square.x][square.y]
        score: float = 0.0

        if opponent_heat < 0.4:
            score -= 80.0
        # if opponent_friendly > opponent_heat: score -= max(30.0, 10.0*(opponent_friendly-opponent_heat))
        # if opponent_heat <1.5: score -=5
        # if opponent_heat > opponent_friendly: score += 10.0*(opponent_friendly-opponent_heat)

        return score


class BotHelper:

    @staticmethod
    def blitz_used(game: g.Game) -> bool:
        for action in game.state.available_actions:
            if action.action_type == t.ActionType.START_BLITZ:
                return False
        return True

    @staticmethod
    def handoff_used(game: g.Game) -> bool:
        for action in game.state.available_actions:
            if action.action_type == t.ActionType.START_HANDOFF:
                return False
        return True

    @staticmethod
    def foul_used(game: g.Game) -> bool:
        for action in game.state.available_actions:
            if action.action_type == t.ActionType.START_FOUL:
                return False
        return True

    @staticmethod
    def pass_used(game: g.Game) -> bool:
        for action in game.state.available_actions:
            if action.action_type == t.ActionType.START_PASS:
                return False
        return True

    @staticmethod
    def get_players(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = True, include_used: bool = True, include_off_pitch: bool = False, only_blockable: bool = False, only_used: bool = False) -> List[m.Player]:
        players: List[m.Player] = []
        selected_players: List[m.Player] = []
        for iteam in game.state.teams:
            if iteam == team and include_own:
                players.extend(iteam.players)
            if iteam != team and include_opp:
                players.extend(iteam.players)
        for player in players:
            if only_blockable and not player.state.up:
                continue
            if only_used and not player.state.used:
                continue

            if include_stunned or not player.state.stunned:
                if include_used or not player.state.used:
                    if include_off_pitch or (player.position is not None and not game.is_out_of_bounds(player.position)):
                        selected_players.append(player)

        return selected_players

    @staticmethod
    def caging_squares_north_east(game: g.Game, protect_square: m.Square) -> List[m.Square]:

        # * At it's simplest, a cage requires 4 players in the North-East, South-East, South-West and North-West
        # * positions, relative to the ball carrier, such that there is no more than 3 squares between the players in
        # * each of those adjacent compass directions.
        # *
        # *   1     3
        # *    xx-xx
        # *    xx-xx
        # *    --o--
        # *    xx-xx
        # *    xx-xx
        # *   3     4
        # *
        # * pitch is 26 long
        # *
        # *
        # * Basically we need one player in each of the corners: 1-4, but spaced such that there is no gap of 3 squares.
        # * If the caging player is in 1-4, but next to ball carrier, he ensures this will automatically be me
        # *
        # * The only exception to this is when the ball carrier is on, or near, the sideline.  Then return the squares
        # * that can otherwise form the cage.
        # *

        caging_squares: List[m.Square] = []
        x = protect_square.x
        y = protect_square.y

        if x <= game.state.pitch.width - 3:
            if y == game.state.pitch.height - 2:
                caging_squares.append(game.get_square(x + 1, y + 1))
                caging_squares.append(game.get_square(x + 2, y + 1))
                caging_squares.append(game.get_square(x + 1, y))
                caging_squares.append(game.get_square(x + 2, y))
            elif y == game.state.pitch.height - 1:
                caging_squares.append(game.get_square(x + 1, y))
                caging_squares.append(game.get_square(x + 2, y))
            else:
                caging_squares.append(game.get_square(x + 1, y + 1))
                caging_squares.append(game.get_square(x + 1, y + 2))
                caging_squares.append(game.get_square(x + 2, y + 1))
                # caging_squares.append(game.state.pitch.get_square(x + 3, y + 3))

        return caging_squares

    @staticmethod
    def caging_squares_north_west(game: g.Game, protect_square: m.Square) -> List[m.Square]:

        caging_squares: List[m.Square] = []
        x = protect_square.x
        y = protect_square.y

        if x >= 3:
            if y == game.state.pitch.height-2:
                caging_squares.append(game.get_square(x - 1, y + 1))
                caging_squares.append(game.get_square(x - 2, y + 1))
                caging_squares.append(game.get_square(x - 1, y))
                caging_squares.append(game.get_square(x - 2, y))
            elif y == game.state.pitch.height-1:
                caging_squares.append(game.get_square(x - 1, y))
                caging_squares.append(game.get_square(x - 2, y))
            else:
                caging_squares.append(game.get_square(x - 1, y + 1))
                caging_squares.append(game.get_square(x - 1, y + 2))
                caging_squares.append(game.get_square(x - 2, y + 1))
                # caging_squares.append(game.state.pitch.get_square(x - 3, y + 3))

        return caging_squares

    @staticmethod
    def caging_squares_south_west(game: g.Game, protect_square: m.Square) -> List[m.Square]:

        caging_squares: List[m.Square] = []
        x = protect_square.x
        y = protect_square.y

        if x >= 3:
            if y == 2:
                caging_squares.append(game.get_square(x - 1, y - 1))
                caging_squares.append(game.get_square(x - 2, y - 1))
                caging_squares.append(game.get_square(x - 1, y))
                caging_squares.append(game.get_square(x - 2, y))
            elif y == 1:
                caging_squares.append(game.get_square(x - 1, y))
                caging_squares.append(game.get_square(x - 2, y))
            else:
                caging_squares.append(game.get_square(x - 1, y - 1))
                caging_squares.append(game.get_square(x - 1, y - 2))
                caging_squares.append(game.get_square(x - 2, y - 1))
                # caging_squares.append(game.state.pitch.get_square(x - 3, y - 3))

        return caging_squares

    @staticmethod
    def caging_squares_south_east(game: g.Game, protect_square: m.Square) -> List[m.Square]:

        caging_squares: List[m.Square] = []
        x = protect_square.x
        y = protect_square.y

        if x <= game.state.pitch.width - 3:
            if y == 2:
                caging_squares.append(game.get_square(x + 1, y - 1))
                caging_squares.append(game.get_square(x + 2, y - 1))
                caging_squares.append(game.get_square(x + 1, y))
                caging_squares.append(game.get_square(x + 2, y))
            elif y == 1:
                caging_squares.append(game.get_square(x + 1, y))
                caging_squares.append(game.get_square(x + 2, y))
            else:
                caging_squares.append(game.get_square(x + 1, y - 1))
                caging_squares.append(game.get_square(x + 1, y - 2))
                caging_squares.append(game.get_square(x + 2, y - 1))
                # caging_squares.append(game.get_square(x + 3, y - 3))

        return caging_squares

    @staticmethod
    def is_caging_position(game: g.Game, player: m.Player, protect_player: m.Player) -> bool:
        return player.position.distance(protect_player.position) <= 2 and not BotHelper.is_castle_position_of(game, player, protect_player)

    @staticmethod
    def has_player_within_n_squares(game: g.Game, units: List[m.Player], square: m.Square, num_squares: int) -> bool:
        for cur in units:
            if cur.position.distance(square) <= num_squares:
                return True
        return False

    @staticmethod
    def has_adjacent_player(game: g.Game, square: m.Square) -> bool:
        return not game.get_adjacent_players(square)

    @staticmethod
    def is_castle_position_of(game: g.Game, player1: m.Player, player2: m.Player) -> bool:
        return player1.position.x == player2.position.x or player1.position.y == player2.position.y

    @staticmethod
    def is_bishop_position_of(game: g.Game, player1: m.Player, player2: m.Player) -> bool:
        return abs(player1.position.x - player2.position.x) == abs(player1.position.y - player2.position.y)

    @staticmethod
    def attacker_would_surf(game: g.Game, attacker: m.Player, defender: m.Player) -> bool:
        if (defender.has_skill(t.Skill.SIDE_STEP) and not attacker.has_skill(t.Skill.GRAB)) or defender.has_skill(t.Skill.STAND_FIRM):
            return False

        if not attacker.position.is_adjacent(defender.position):
            return False

        return BotHelper.direct_surf_squares(game, attacker.position, defender.position)

    @staticmethod
    def direct_surf_squares(game: g.Game, attack_square: m.Square, defend_square: m.Square) -> bool:
        defender_on_sideline: bool = BotHelper.on_sideline(game, defend_square)
        defender_in_endzone: bool = BotHelper.on_endzone(game, defend_square)

        if defender_on_sideline and defend_square.x == attack_square.x:
            return True

        if defender_in_endzone and defend_square.y == attack_square.y:
            return True

        if defender_in_endzone and defender_on_sideline:
            return True

        return False

    @staticmethod
    def reverse_x_for_right(game: g.Game, team: m.Team, x: int) -> int:
        if not game.is_team_side(m.Square(13, 3), team):
            res = game.state.pitch.width - 1 - x
        else:
            res = x
        return res

    @staticmethod
    def reverse_x_for_left(game: g.Game, team: m.Team, x: int) -> int:
        if game.is_team_side(m.Square(13, 3), team):
            res = game.state.pitch.width - 1 - x
        else:
            res = x
        return res

    @staticmethod
    def on_sideline(game: g.Game, square: m.Square) -> bool:
        return square.y == 1 or square.y == game.state.pitch.height - 1

    @staticmethod
    def on_endzone(game: g.Game, square: m.Square) -> bool:
        return square.x == 1 or square.x == game.state.pitch.width - 1

    @staticmethod
    def on_los(game: g.Game, team: m.Team, square: m.Square) -> bool:
        return (BotHelper.reverse_x_for_right(game, team, square.x) == 13) and 4 < square.y < 21

    @staticmethod
    def los_squares(game: g.Game, team: m.Team) -> List[m.Square]:

        squares: List[m.Square] = [
            game.get_square(BotHelper.reverse_x_for_right(game, team, 13), 5),
            game.get_square(BotHelper.reverse_x_for_right(game, team, 13), 6),
            game.get_square(BotHelper.reverse_x_for_right(game, team, 13), 7),
            game.get_square(BotHelper.reverse_x_for_right(game, team, 13), 8),
            game.get_square(BotHelper.reverse_x_for_right(game, team, 13), 9),
            game.get_square(BotHelper.reverse_x_for_right(game, team, 13), 10),
            game.get_square(BotHelper.reverse_x_for_right(game, team, 13), 11)
        ]
        return squares

    @staticmethod
    def distance_to_sideline(game: g.Game, square: m.Square) -> int:
        return min(square.y - 1, game.state.pitch.height - square.y - 2)

    @staticmethod
    def is_endzone(game, square: m.Square) -> bool:
        return square.x == 1 or square.x == game.state.pitch.width - 1

    @staticmethod
    def last_block_proc(game) -> Optional[p.Block]:
        for i in range(len(game.state.stack.items) - 1, -1, -1):
            if isinstance(game.state.stack.items[i], p.Block):
                block_proc = game.state.stack.items[i]
                return block_proc
        return None

    @staticmethod
    def is_adjacent_ball(game: g.Game, square: m.Square) -> bool:
        ball_square = game.get_ball_position()
        return ball_square is not None and ball_square.is_adjacent(square)

    @staticmethod
    def squares_within(game: g.Game, square: m.Square, distance: int) -> List[m.Square]:
        squares: List[m.Square] = []
        for i in range(-distance, distance + 1):
            for j in range(-distance, distance + 1):
                cur_square = game.get_square(square.x + i, square.y + j)
                if cur_square != square and not game.is_out_of_bounds(cur_square):
                    squares.append(cur_square)
        return squares

    @staticmethod
    def distance_to_defending_endzone(game: g.Game, team: m.Team, position: m.Square) -> int:
        res = BotHelper.reverse_x_for_right(game, team, position.x) - 1
        return res

    @staticmethod
    def distance_to_scoring_endzone(game: g.Game, team: m.Team, position: m.Square) -> int:
        res = BotHelper.reverse_x_for_left(game, team, position.x) - 1
        return res
        # return game.state.pitch.width - 1 - reverse_x_for_right(game, team, position.x)

    @staticmethod
    def players_in_scoring_endzone(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = False) -> List[m.Player]:
        players: List[m.Player] = BotHelper.get_players(game, team, include_own=include_own, include_opp=include_opp)
        selected_players: List[m.Player] = []
        for player in players:
            if BotHelper.in_scoring_endzone(game, team, player.position):
                selected_players.append(player)
        return selected_players

    @staticmethod
    def in_scoring_endzone(game: g.Game, team: m.Team, square: m.Square) -> bool:
        return BotHelper.reverse_x_for_left(game, team, square.x) == 1

    @staticmethod
    def players_in_scoring_distance(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = False) -> List[m.Player]:
        players: List[m.Player] = BotHelper.get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned)
        selected_players: List[m.Player] = []
        for player in players:
            if BotHelper.distance_to_scoring_endzone(game, team, player.position) <= player.num_moves_left():
                selected_players.append(player)
        return selected_players

    @staticmethod
    def distance_to_nearest_player(game: g.Game, team: m.Team, square: m.Square, include_own: bool = True, include_opp: bool = True, only_used: bool = False, include_used: bool = True, include_stunned: bool = True, only_blockable: bool = False) -> int:
        opps: List[m.Player] = BotHelper.get_players(game, team, include_own=include_own, include_opp=include_opp, only_used=only_used, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)
        cur_max = 100
        for opp in opps:
            dist = opp.position.distance(square)
            cur_max = min(cur_max, dist)
        return cur_max

    @staticmethod
    def screening_distance(game: g.Game, from_square: m.Square, to_square: m.Square) -> float:
        # Return the "screening distance" between 3 squares.  (To complete)
        # float dist =math.sqrt(math.pow(m.Square.x - cur.position.x, 3) + math.pow(m.Square.y - cur.position.y, 3))
        return 0.0

    @staticmethod
    def num_opponents_can_reach(game: g.Game, team: m.Team, square: m.Square) -> int:
        opps: List[m.Player] = BotHelper.get_players(game, team, include_own=False, include_opp=True)
        num_opps_reach: int = 0
        for cur in opps:
            dist = max(square.x - cur.position.x, square.y - cur.position.y)
            if cur.state.stunned:
                continue
            move_allowed = cur.get_ma() + 2
            if not cur.state.up:
                move_allowed -= 3
            if dist < move_allowed:
                num_opps_reach += 1
        return num_opps_reach

    @staticmethod
    def num_opponents_on_field(game: g.Game, team: m.Team) -> int:
        opps: List[m.Player] = BotHelper.get_players(game, team, include_own=False, include_opp=True)
        num_opponents = 0
        for cur in opps:
            if cur.position is not None:
                num_opponents += 1
        return num_opponents

    @staticmethod
    def number_opponents_closer_than_to_endzone(game: g.Game, team: m.Team, square: m.Square) -> int:
        opponents: List[m.Player] = BotHelper.get_players(game, team, include_own=False, include_opp=True)
        num_opps = 0
        distance_square_endzone = BotHelper.distance_to_defending_endzone(game, team, square)

        for opponent in opponents:
            distance_opponent_endzone = BotHelper.distance_to_defending_endzone(game, team, opponent.position)
            if distance_opponent_endzone < distance_square_endzone:
                num_opps += 1
        return num_opps

    @staticmethod
    def in_scoring_range(game: g.Game, player: m.Player) -> bool:
        return player.num_moves_left() >= BotHelper.distance_to_scoring_endzone(game, player.team, player.position)

    @staticmethod
    def players_in_scoring_range(game: g.Game, team: m.Team, include_own=True, include_opp=True, include_used=True, include_stunned=True) -> List[m.Player]:
        players: List[m.Player] = BotHelper.get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned, include_used=include_used)
        res: List[m.Player] = []
        for player in players:
            if BotHelper.in_scoring_range(game, player):
                res.append(player)
        return res

    @staticmethod
    def players_in(game: g.Game, team: m.Team, squares: List[m.Square], include_own=True, include_opp=True, include_used=True, include_stunned=True, only_blockable=False) -> List[m.Player]:

        allowed_players: List[m.Player] = BotHelper.get_players(game, team, include_own=include_own, include_opp=include_opp, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)
        res: List[m.Player] = []

        for square in squares:
            player: Optional[m.Player] = game.get_player_at(square)
            if player is None:
                continue
            if player in allowed_players:
                res.append(player)
        return res

    @staticmethod
    def block_favourability(block_result: m.ActionType, team: m.Team, active_player: m.Player, attacker: m.Player, defender: m.Player, favor: m.Team) -> float:

        if attacker.team == active_player.team:
            if block_result == t.ActionType.SELECT_DEFENDER_DOWN:
                return 6.0
            elif block_result == t.ActionType.SELECT_DEFENDER_STUMBLES:
                if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE):
                    return 4.0       # push back
                else:
                    return 6.0
            elif block_result == t.ActionType.SELECT_PUSH:
                return 4.0
            elif block_result == t.ActionType.SELECT_BOTH_DOWN:
                if attacker.has_skill(t.Skill.BLOCK):
                    if defender.has_skill(t.Skill.BLOCK):
                        # Nothing happens
                        return 3.0
                    else:
                        # Defender down
                        return 5.0
                else:
                    if defender.has_skill(t.Skill.BLOCK):
                        # Only defender down
                        return 1.0
                    else:
                        # Both down
                        return 2.0
                                                                                # only defender is down
            elif block_result == t.ActionType.SELECT_ATTACKER_DOWN:
                return 1.0                                                                                        # skull
        else:
            if block_result == t.ActionType.SELECT_DEFENDER_DOWN:
                return 1.0                                                                                        # least favourable
            elif block_result == t.ActionType.SELECT_DEFENDER_STUMBLES:
                if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE):
                    return 3       # not going down, so I like this.
                else:
                    return 1.0                                                                                  # splat.  No good.
            elif block_result == t.ActionType.SELECT_PUSH:
                return 3.0
            elif block_result == t.ActionType.SELECT_BOTH_DOWN:
                if not attacker.has_skill(t.Skill.BLOCK) and defender.has_skill(t.Skill.BLOCK):
                    return 6.0        # Attacker down, I am not.
                if not attacker.has_skill(t.Skill.BLOCK) and not defender.has_skill(t.Skill.BLOCK):
                    return 5.0    # Both down is pretty good.
                if attacker.has_skill(t.Skill.BLOCK) and not defender.has_skill(t.Skill.BLOCK):
                    return 2.0        # Just I splat
                else:
                    return 4.0                                                                                  # Nothing happens (both have block).
            elif block_result == t.ActionType.SELECT_ATTACKER_DOWN:
                return 6.0                                                                                        # most favourable!

        return 0.0

    @staticmethod
    def potential_end_player_turn_action(game: g.Game, heat_map, player: m.Player) -> List[ActionSequence]:
        actions: List[ActionSequence] = []
        action_steps: List[m.Action] = [
            m.Action(t.ActionType.END_PLAYER_TURN)
            ]
        # End turn happens on a score of 1.0.  Any actions with a lower score are never selected.
        actions.append(ActionSequence(action_steps, score=1.0, description='End Turn', player=player))
        return actions

    @staticmethod
    def potential_end_turn_action(game: g.Game) -> List[ActionSequence]:
        actions: List[ActionSequence] = []
        action_steps: List[m.Action] = [
            m.Action(t.ActionType.END_TURN)
            ]
        # End turn happens on a score of 1.0.  Any actions with a lower score are never selected.
        actions.append(ActionSequence(action_steps, score=1.0, description='End Turn'))
        return actions

    @staticmethod
    def potential_block_actions(game: g.Game, heat_map: FfHeatMap, player: m.Player) -> List[ActionSequence]:

        # Note to self: need a "stand up and end move option.
        move_actions: List[ActionSequence] = []
        if not player.state.up:
            # There is currently a bug in the controlling logic.  Prone players shouldn't be able to block
            return move_actions
        blockable_players: List[m.Player] = game.get_adjacent_opponents(player, standing=True, stunned=False, down=False)
        for blockable_player in blockable_players:
            action_steps: List[m.Action] = [
                m.Action(t.ActionType.START_BLOCK, player=player),
                m.Action(t.ActionType.BLOCK, position=blockable_player.position),
                m.Action(t.ActionType.END_PLAYER_TURN)
            ]

            action_score = BotHelper.score_block(game, heat_map, player, blockable_player)
            score = action_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Block ' + player.name + ' to (' + str(blockable_player.position.x) + ',' + str(blockable_player.position.y) + ')', player=player))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
        return move_actions

    @staticmethod
    def potential_blitz_actions(game: g.Game, heat_map: FfHeatMap, player: m.Player, path: pf.Path) -> List[ActionSequence]:
        move_actions: List[ActionSequence] = []     
        from_position = path.steps[-2] if len(path.steps)>1 else player.position
        to_square = path.steps[-1]

        action_score = BotHelper.score_blitz(game, heat_map, player, from_position, game.get_player_at(to_square))
        path_score = BotHelper.path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
        score = action_score + path_score

        action_steps: List[Action] = []
        action_steps.append(Action(ActionType.START_BLITZ, player=player))
        action_steps.extend(BotHelper.path_to_move_actions(game, player, path))
        move_actions.append(ActionSequence(action_steps, score=score, description='Blitz ' + player.name + ' to ' + str(to_square.x) + ',' + str(to_square.y), player=player))

        # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
        return move_actions

    @staticmethod
    def path_to_move_actions(game: botbowl.Game, player: botbowl.Player, path: pf.Path, do_assertions=True) -> List[Action]:
        """
        This function converts a path into a list of actions corresponding to that path.
        If you provide a handoff, foul or blitz path, then you have to manally set action type.
        :param game:
        :param player: player to move
        :param path: a path as returned by the pathfinding algorithms
        :param do_assertions: if False, it turns off the validation, can be helpful when the GameState will change before
                            this path is executed.
        :returns: List of actions corresponding to 'path'.
        """

        if path.block_dice is not None:
            action_type = ActionType.BLOCK
        elif path.handoff_roll is not None:
            action_type = ActionType.HANDOFF
        elif path.foul_roll is not None:
            action_type = ActionType.FOUL
        else:
            action_type = ActionType.MOVE

        active_team = game.state.available_actions[0].team
        player_at_target = game.get_player_at(path.get_last_step())

        if do_assertions:
            if action_type is ActionType.MOVE:
                assert player_at_target is None or player_at_target is game.get_active_player()
            elif action_type is ActionType.BLOCK:
                assert game.get_opp_team(active_team) is player_at_target.team
                assert player_at_target.state.up
            elif action_type is ActionType.FOUL:
                assert game.get_opp_team(active_team) is player_at_target.team
                assert not player_at_target.state.up
            elif action_type is ActionType.HANDOFF:
                assert active_team is player_at_target.team
                assert player_at_target.state.up
            else:
                raise Exception(f"Unregonized action type {action_type}")

        final_action = Action(action_type, position=path.get_last_step())

        if game._is_action_allowed(final_action):
            return [final_action]
        else:
            actions = []
            if not player.state.up and path.steps[0] == player.position:
                actions.append(Action(ActionType.STAND_UP, player=player))
                actions.extend(Action(ActionType.MOVE, position=sq) for sq in path.steps[1:-1])
            else:
                actions.extend(Action(ActionType.MOVE, position=sq) for sq in path.steps[:-1])
            actions.append(final_action)
            return actions

    @staticmethod
    def potential_pass_actions(game: g.Game, heat_map: FfHeatMap, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
        move_actions: List[ActionSequence] = []
        for path in paths:
            path_steps = path.steps
            end_square: m.Square = game.get_square(path.get_last_step().x, path.get_last_step().y)
            # Need possible receving players
            to_squares, distances = game.get_pass_distances_at(player, player, end_square)
            for to_square in to_squares:
                action_steps = [m.Action(t.ActionType.START_PASS, player=player)]

                receiver: Optional[m.Player] = game.get_player_at(to_square)

                if not player.state.up:
                    action_steps.append(m.Action(t.ActionType.STAND_UP))
                for step in path_steps:
                    if step == player.position:
                        continue    # if down might have step in home square
                    # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                    action_steps.append(m.Action(t.ActionType.MOVE, position=step))
                action_steps.append(m.Action(t.ActionType.PASS, position=to_square))
                action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN))

                action_score = BotHelper.score_pass(game, heat_map, player, end_square, to_square)
                path_score = BotHelper.path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
                score = action_score + path_score

                move_actions.append(ActionSequence(action_steps, score=score, description='Pass ' + player.name + ' to ' + str(to_square.x) + ',' + str(to_square.y), player=player))
                # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
        return move_actions

    @staticmethod
    def potential_handoff_actions(game: g.Game, heat_map: FfHeatMap, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
        move_actions: List[ActionSequence] = []
        for path in paths:
            path_steps = path.steps
            end_square: m.Square = game.get_square(path.get_last_step().x, path.get_last_step().y)
            handoffable_players = game.get_adjacent_players(end_square, team=player.team, standing=True, down=False, stunned=False)
            for handoffable_player in handoffable_players:
                action_steps: List[m.Action] = [m.Action(t.ActionType.START_HANDOFF, player=player)]
                for step in path_steps:
                    if step == player.position:
                        continue    # if down might have step in home square
                    # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                    action_steps.append(m.Action(t.ActionType.MOVE, position=step))
                action_steps.append(m.Action(t.ActionType.HANDOFF, position=handoffable_player.position))
                action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN))

                action_score = BotHelper.score_handoff(game, heat_map, player, handoffable_player, end_square)
                path_score = BotHelper.path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
                score = action_score + path_score

                move_actions.append(ActionSequence(action_steps, score=score, description='Handoff ' + player.name + ' to ' + str(handoffable_player.position.x) + ',' + str(handoffable_player.position.y), player=player))
                # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
        return move_actions

    @staticmethod
    def potential_foul_actions(game: g.Game, heat_map: FfHeatMap, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
        move_actions: List[ActionSequence] = []
        for path in paths:
            path_steps = path.steps
            end_square: m.Square = game.get_square(path.get_last_step().x, path.get_last_step().y)
            foulable_players = game.get_adjacent_players(end_square, team=game.get_opp_team(player.team),  standing=False, stunned=True, down=True)
            for foulable_player in foulable_players:
                action_steps: List[m.Action] = [m.Action(t.ActionType.START_FOUL, player=player)]
                if not player.state.up:
                    action_steps.append(m.Action(t.ActionType.STAND_UP))
                for step in path_steps:
                    if step == player.position:
                        continue    # if down might have step in home square
                    # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                    action_steps.append(m.Action(t.ActionType.MOVE, position=step))
                action_steps.append(m.Action(t.ActionType.FOUL, foulable_player.position))
                action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN))

                action_score = BotHelper.score_foul(game, heat_map, player, foulable_player, end_square)
                path_score = BotHelper.path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
                score = action_score + path_score

                move_actions.append(ActionSequence(action_steps, score=score, description='Foul ' + player.name + ' to ' + str(foulable_player.position.x) + ',' + str(foulable_player.position.y), player=player))
                # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
        return move_actions

    @staticmethod
    def potential_move_actions(game: g.Game, heat_map: FfHeatMap, player: m.Player, paths: List[pf.Path], is_continuation: bool = False) -> List[ActionSequence]:

        move_actions: List[ActionSequence] = []
        ball_square: m.Square = game.get_ball_position()
        if not player.has_tackle_zone():
            # consider standing and doing nothing
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(t.ActionType.START_MOVE, player=player))
            action_steps.append(m.Action(t.ActionType.STAND_UP))
            action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN))
            score, is_complete, description = BotHelper.score_move(game, heat_map, player, player.position)
            action = ActionSequence(action_steps, score=score, description=f'''Stand Up: {description} {player.name} {player.position.x}, {player.position.y}''', player=player)
            move_actions.append(action)

        for path in paths:
            path_steps = path.steps
            action_steps: List[m.Action] = []
            if not is_continuation:
                action_steps.append(m.Action(t.ActionType.START_MOVE, player=player))
            if not player.state.up:
                #if path_steps[0] == player.position:
                #    del path_steps[0]
                action_steps.append(m.Action(t.ActionType.STAND_UP))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                if step == player.position:
                    continue    # path for standing up included stating square?
                action_steps.append(m.Action(t.ActionType.MOVE, position=step))

            to_square: m.Square = path.get_last_step()
            action_score, is_complete, description = BotHelper.score_move(game, heat_map, player, to_square)
            if is_complete:
                action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN))

            if is_continuation:
                # Continuing actions (after a Blitz block for example) may choose risky options, so penalise risk
                path_score = BotHelper.continuation_path_cost_to_score(path)

            else:
                path_score = BotHelper.path_cost_to_score(path)
            score = action_score + path_score
            action = ActionSequence(action_steps, score=score, description=f'''Move: {description} {player.name} {player.position.x}, {player.position.y} to {path.get_last_step().x}, {path.get_last_step().y}''', player=player)

            move_actions.append(action)
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
        return move_actions

    @staticmethod
    def score_blitz(game: g.Game, heat_map: FfHeatMap, attacker: m.Player, block_from_square: m.Square, defender: m.Player) -> float:
        score: float = GrodBot.BASE_SCORE_BLITZ

        ball_carrier: Optional[m.Player] = game.get_ball_carrier()
        is_ball_carrier = attacker == ball_carrier
        defender_is_ball_carrier = defender == ball_carrier

        num_block_dice: int = game.num_block_dice_at(attacker, defender, block_from_square, blitz=True, dauntless_success=False)
        ball_position: m.Player = game.get_ball_position()
        if num_block_dice == 3:
            score += 30.0
        if num_block_dice == 2:
            score += 10.0
        if num_block_dice == 1:
            score += -30.0
        if num_block_dice == -2:
            score += -75.0
        if num_block_dice == -3:
            score += -100.0
        if attacker.has_skill(t.Skill.BLOCK):
            score += 20.0
        if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE):
            score -= 10.0
        if defender.has_skill(t.Skill.BLOCK):
            score += -10.0
        if is_ball_carrier:
            if attacker.position.is_adjacent(defender.position) and block_from_square == attacker.position:
                score += 20.0  # Favour blitzing with ball carrier at start of move
            else:
                score += -40.0  # But don't blitz with ball carrier after that
        if defender_is_ball_carrier:
            score += 55.0              # Blitzing ball carrier
        else:
            score -= 10
        if defender.position.is_adjacent(ball_position):
            score += 20.0   # Blitzing someone adjacent to ball carrier
        if BotHelper.direct_surf_squares(game, block_from_square, defender.position):
            score += 25.0  # A surf
        if game.get_adjacent_opponents(attacker, stunned=False, down=False) and not is_ball_carrier:
            score -= 10.0
        if attacker.position == block_from_square:
            score -= 20.0   # A Blitz where the block is the starting square is unattractive
        if BotHelper.in_scoring_range(game, defender):
            score += 10.0  # Blitzing players closer to the endzone is attractive
        return score

    @staticmethod
    def score_foul(game: g.Game, heat_map: FfHeatMap, attacker: m.Player, defender: m.Player, to_square: m.Square) -> float:
        score = GrodBot.BASE_SCORE_FOUL
        ball_carrier: Optional[m.Player] = game.get_ball_carrier()

        if ball_carrier == attacker:
            score = score - 30.0
        if attacker.has_skill(t.Skill.DIRTY_PLAYER):
            score = score + 10.0
        if attacker.has_skill(t.Skill.SNEAKY_GIT):
            score = score + 10.0
        if defender.state.stunned:
            score = score - 15.0

        assists_for, assists_against = game.num_assists_at(attacker, defender, to_square, foul=True)
        score = score + (assists_for-assists_against) * 15.0

        if attacker.team.state.bribes > 0:
            score += 40.0
        if attacker.has_skill(t.Skill.CHAINSAW):
            score += 30.0
        # TVdiff = defender.GetBaseTV() - attacker.GetBaseTV()
        tv_diff = 10.0
        score = score + tv_diff

        return score

    @staticmethod
    def score_move(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool, str]:

        scores: List[(float, bool, str)] = [
            [*BotHelper.score_receiving_position(game, heat_map, player, to_square), 'move to receiver'],
            [*BotHelper.score_move_to_ball(game, heat_map, player, to_square), 'move to ball'],
            [*BotHelper.score_move_towards_ball(game, heat_map, player, to_square), 'move toward ball'],
            [*BotHelper.score_move_ball(game, heat_map, player, to_square), 'move ball'],
            [*BotHelper.score_sweep(game, heat_map, player, to_square), 'move to sweep'],
            [*BotHelper.score_defensive_screen(game, heat_map, player, to_square), 'move to defensive screen'],
            [*BotHelper.score_offensive_screen(game, heat_map, player, to_square), 'move to offsensive screen'],
            [*BotHelper.score_caging(game, heat_map, player, to_square), 'move to cage'],
            [*BotHelper.score_mark_opponent(game, heat_map, player, to_square), 'move to mark opponent']
            ]

        scores.sort(key=lambda tup: tup[0], reverse=True)
        score, is_complete, description = scores[0]

        # All moves should avoid the sideline
        if BotHelper.distance_to_sideline(game, to_square) == 0:
            score += GrodBot.ADDITIONAL_SCORE_SIDELINE
        if BotHelper.distance_to_sideline(game, to_square) == 1:
            score += GrodBot.ADDITIONAL_SCORE_NEAR_SIDELINE

        if not player.state.up:
            score += GrodBot.ADDITIONAL_SCORE_PRONE

        if to_square == player.position:
            x = 1

        return score, is_complete, description

    @staticmethod
    def score_receiving_position(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:
        ball_carrier = game.get_ball_carrier()
        if ball_carrier is not None and (player.team != ball_carrier.team or player == ball_carrier):
            return 0.0, True

        receivingness = BotHelper.player_receiver_ability(game, player)
        score = receivingness - 30.0
        if BotHelper.in_scoring_endzone(game, player.team, to_square):
            num_in_range = len(BotHelper.players_in_scoring_endzone(game, player.team, include_own=True, include_opp=False))
            if player.team.state.turn == 8:
                score += 40   # Pretty damned urgent to get to end zone!
            score -= num_in_range * num_in_range * 40  # Don't want too many catchers in the endzone ...

        score += 5.0 * (max(BotHelper.distance_to_scoring_endzone(game, player.team, player.position), player.get_ma()) - max(BotHelper.distance_to_scoring_endzone(game, player.team, to_square), player.get_ma()))
        # Above score doesn't push players to go closer than their MA from the endzone.

        if BotHelper.distance_to_scoring_endzone(game, player.team, to_square) > player.get_ma() + 2:
            score -= 30.0
        opp_team = game.get_opp_team(player.team)
        opps: List[m.Player] = game.get_adjacent_players(player.position, opp_team, stunned=False, down=False)
        if opps:
            score -= 40.0 + 20.0 * len(opps)
        score -= 10.0 * len(game.get_adjacent_players(to_square, opp_team, stunned=False, down=False))
        num_in_range = len(BotHelper.players_in_scoring_distance(game, player.team, include_own=True, include_opp=False))
        score -= num_in_range * num_in_range * 20.0     # Lower the score if we already have some receivers.
        if BotHelper.players_in(game, player.team, BotHelper.squares_within(game, to_square, 2), include_opp=False, include_own=True):
            score -= 20.0

        return score, True

    @staticmethod
    def score_move_towards_ball(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:
        ball_square: m.Square = game.get_ball_position()
        ball_carrier = game.get_ball_carrier()
        if ball_carrier is not None:
            ball_team = ball_carrier.team
        else:
            ball_team = None

        if (to_square == ball_square) or ((ball_team is not None) and (ball_team == player.team)):
            return 0.0, True

        score = GrodBot.BASE_SCORE_MOVE_TOWARD_BALL
        if ball_carrier is None:
            score += 20.0

        if ball_square is not None:
            player_distance_to_ball = ball_square.distance(player.position)
            destination_distance_to_ball = ball_square.distance(to_square)
            score += (player_distance_to_ball - destination_distance_to_ball)

            if destination_distance_to_ball <= 2:
                score += 10
            elif destination_distance_to_ball <= 3:
                score += 5
            else:        # Need an anti-crowding measure
                d = BotHelper.distance_to_nearest_player(game, player.team, to_square, include_own=True, include_opp=False, only_blockable=True)

                if d == 1:
                    score -= 15
                elif d == 2:
                    score -= 10
                elif d == 3:
                    score -= 5

        return score, True

    @staticmethod
    def score_move_to_ball(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:
        ball_square: m.Square = game.get_ball_position()
        ball_carrier = game.get_ball_carrier()
        if (ball_square != to_square) or (ball_carrier is not None):
            return 0.0, True

        score = GrodBot.BASE_SCORE_MOVE_TO_BALL
        if player.has_skill(t.Skill.SURE_HANDS) or not player.team.state.reroll_used:
            score += 15.0
        elif not player.team.state.reroll_used:
            score += 10.0
        if player.get_ag() < 2:
            score += -10.0
        if player.get_ag() == 3:
            score += 5.0
        if player.get_ag() > 3:
            score += 10.0
        num_tz = game.num_tackle_zones_at(player, ball_square)
        score += - 10 * num_tz    # Lower score if lots of tackle zones on ball.

        # If there is only 1 or 3 players left to move, lets improve score of trying to pick the ball up
        players_to_move: List[m.Player] = BotHelper.get_players(game, player.team, include_own=True, include_opp=False, include_used=False, include_stunned=False)
        if len(players_to_move) == 1:
            score += 25
        if len(players_to_move) == 2:
            score += 15

        # If the current player is the best player to pick up the ball, increase the score
        players_sorted_blitz = sorted(players_to_move, key=lambda x: BotHelper.player_blitz_ability(game, x), reverse=True)
        if players_sorted_blitz[0] == player:
            score += 5
        players_sorted_passing = sorted(players_to_move, key=lambda x: BotHelper.player_pass_ability(game, x), reverse=True)
        if players_sorted_passing[0] == player:
            score += 9

        # Cancel the penalty for being near the sideline if the ball is on/near the sideline (it's applied later)
        if BotHelper.distance_to_sideline(game, ball_square) == 1:
            score -= GrodBot.ADDITIONAL_SCORE_NEAR_SIDELINE
        if BotHelper.distance_to_sideline(game, ball_square) == 0:
            score -= GrodBot.ADDITIONAL_SCORE_SIDELINE

        # Need to increase score if no other player is around to get the ball (to do)

        return score, False

    @staticmethod
    def score_move_ball(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:
        # ball_square: m.Square = game.get_ball_position()
        ball_carrier = game.get_ball_carrier()
        if (ball_carrier is None) or player != ball_carrier:
            return 0.0, True

        score = GrodBot.BASE_SCORE_MOVE_BALL
        if BotHelper.in_scoring_endzone(game, player.team, to_square):
            if player.team.state.turn == 8:
                score += 115.0  # Make overwhelmingly attractive
            else:
                score += 60.0  # Make scoring attractive
        elif player.team.state.turn == 8:
            score -= 100.0  # If it's the last turn, heavily penalyse a non-scoring action
        else:
            score += heat_map.get_ball_move_square_safety_score(to_square)
            opps: List[m.Player] = game.get_adjacent_players(to_square, team=game.get_opp_team(player.team), stunned=False)
            if opps:
                score -= (40.0 + 20.0 * len(opps))
            opps_close_to_destination = BotHelper.players_in(game, player.team, BotHelper.squares_within(game, to_square, 2), include_own=False, include_opp=True, include_stunned=False)
            if opps_close_to_destination:
                score -= (20.0 + 5.0 * len(opps_close_to_destination))
            if not BotHelper.blitz_used(game):
                score -= 30.0  # Lets avoid moving the ball until the Blitz has been used (often helps to free the move).

            dist_player = BotHelper.distance_to_scoring_endzone(game, player.team, player.position)
            dist_destination = BotHelper.distance_to_scoring_endzone(game, player.team, to_square)
            score += 8.0 * (dist_player - dist_destination)  # Increase score the closer we get to the scoring end zone

            # Try to keep the ball central
            if BotHelper.distance_to_sideline(game, to_square) < 3:
                score -= 30

        return score, True

    @staticmethod
    def score_sweep(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:
        ball_carrier = game.get_ball_carrier()
        opposing_team = game.get_opp_team(player.team)
        if ball_carrier is not None:
            ball_team = ball_carrier.team
        else:
            ball_team = None
        if ball_team != opposing_team:
            return 0.0, True  # Don't sweep unless the other team has the ball
        if BotHelper.distance_to_defending_endzone(game, player.team, game.get_ball_position()) < 9:
            return 0.0, True  # Don't sweep when the ball is close to the endzone
        if BotHelper.players_in_scoring_distance(game, player.team, include_own=False, include_opp=True):
            return 0.0, True  # Don't sweep when there are opponent units in scoring range

        score = GrodBot.BASE_SCORE_MOVE_TO_SWEEP
        blitziness = BotHelper.player_blitz_ability(game, player)
        score += blitziness - 60.0
        score -= 30.0 * len(game.get_adjacent_opponents(player, standing=True, down=False, stunned=False))

        # Now to evaluate ideal square for Sweeping:

        x_preferred = int(BotHelper.reverse_x_for_left(game, player.team, (game.state.pitch.width-2) / 4))
        y_preferred = int((game.state.pitch.height-2) / 2)
        score -= abs(y_preferred - to_square .y) * 10.0

        # subtract 5 points for every square away from the preferred sweep location.
        score -= abs(x_preferred - to_square .x) * 5.0

        # Check if a player is already sweeping:
        for i in range(-2, 3):
            for j in range(-2, 3):
                cur: m.Square = game.get_square(x_preferred + i, y_preferred + j)
                player: Optional[m.Player] = game.get_player_at(cur)
                if player is not None and player.team == player.team:
                    score -= 90.0

        return score, True

    @staticmethod
    def score_defensive_screen(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:
        ball_square = game.get_ball_position()
        ball_carrier = game.get_ball_carrier()
        if ball_carrier is not None:
            ball_team = ball_carrier.team
        else:
            ball_team = None

        if ball_team is None or ball_team == player.team:
            return 0.0, True  # Don't screen if we have the ball or ball is on the ground

        # This one is a bit trickier by nature, because it involves combinations of two or more players...
        #    Increase score if square is close to ball carrier.
        #    Decrease if far away.
        #    Decrease if square is behind ball carrier.
        #    Increase slightly if square is 1 away from sideline.
        #    Decrease if close to a player on the same team WHO IS ALREADY screening.
        #    Increase slightly if most of the players movement must be used to arrive at the screening square.

        score = GrodBot.BASE_SCORE_DEFENSIVE_SCREEN

        if ball_square is not None:
            distance_ball_carrier_to_end = BotHelper.distance_to_defending_endzone(game, player.team, ball_square)
            distance_square_to_end = BotHelper.distance_to_defending_endzone(game, player.team, to_square)

            if distance_square_to_end + 1.0 < distance_ball_carrier_to_end:
                score += 30.0  # Increase score defending on correct side of field.

            distance_to_ball = ball_square.distance(to_square)
            score += 8.0*max(4.0 - distance_to_ball, 0.0)  # Increase score defending in front of ball carrier

            score -= 1.0 * (abs(ball_square.y - to_square.y))  # Penalise for defending away from directly in front of ball

            score += distance_square_to_end/10.0  # Increase score a small amount to screen closer to opponents.
        distance_to_closest_opponent = BotHelper.distance_to_nearest_player(game, player.team, to_square, include_own=False, include_opp=True, only_blockable=True)
        if distance_to_closest_opponent <= 1.5:
            score -= 30.0
        elif distance_to_closest_opponent <= 2.95:
            score += 10.0
        elif distance_to_closest_opponent > 2.95:
            score -= 5.0
        if BotHelper.distance_to_sideline(game, to_square) == 1:
            score -= GrodBot.ADDITIONAL_SCORE_NEAR_SIDELINE  # Cancel the negative score of being 1 from sideline.

        distance_to_closest_friendly_used = BotHelper.distance_to_nearest_player(game, player.team, to_square, include_own=True, include_opp=False, only_used=True, only_blockable=True)
        if distance_to_closest_friendly_used >= 4:
            score += 2.0
        elif distance_to_closest_friendly_used >= 3:
            score += 40.0  # Increase score if the square links with another friendly (hopefully also screening)
        elif distance_to_closest_friendly_used > 2:
            score += 10.0   # Descrease score if very close to another defender
        else:
            score -= 10.0  # Decrease score if too close to another defender.

        return score, True

    @staticmethod
    def score_offensive_screen(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:

        # Another subtle one.  Basically if the ball carrier "breaks out", I want to screen him from
        # behind, rather than cage him.  I may even want to do this with an important receiver.
        #     Want my players to be 3 squares from each other, not counting direct diagonals.
        #     Want my players to be hampering the movement of opponent ball or players.
        #     Want my players in a line between goal line and opponent.
        #

        ball_carrier: m.Player = game.get_ball_carrier()
        ball_square: m.Player = game.get_ball_position()
        if ball_carrier is None or ball_carrier.team != player.team:
            return 0.0, True

        score = 0.0     # Placeholder - not implemented yet.

        return score, True

    @staticmethod
    def score_caging(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:
        ball_carrier: m.Player = game.get_ball_carrier()
        if ball_carrier is None or ball_carrier.team != player.team or ball_carrier == player:
            return 0.0, True          # Noone has the ball.  Don't try to cage.
        ball_square: m.Square = game.get_ball_position()

        cage_square_groups: List[List[m.Square]] = [
            BotHelper.caging_squares_north_east(game, ball_square),
            BotHelper.caging_squares_north_west(game, ball_square),
            BotHelper.caging_squares_south_east(game, ball_square),
            BotHelper.caging_squares_south_west(game, ball_square)
            ]

        dist_opp_to_ball = BotHelper.distance_to_nearest_player(game, player.team, ball_square, include_own=False, include_opp=True, include_stunned=False)
        avg_opp_ma = BotHelper.average_ma(game, BotHelper.get_players(game, player.team, include_own=False, include_opp=True, include_stunned=False))

        for curGroup in cage_square_groups:
            if to_square in curGroup and not BotHelper.players_in(game, player.team, curGroup, include_opp=False, include_own=True, only_blockable=True):
                # Test square is inside the cage corner and no player occupies the corner
                score = GrodBot.BASE_SCORE_CAGE_BALL
                dist = BotHelper.distance_to_nearest_player(game, player.team, to_square, include_own=False, include_stunned=False, include_opp=True)
                score += dist_opp_to_ball - dist
                if dist_opp_to_ball > avg_opp_ma:
                    score -= 30.0
                if not ball_carrier.state.used:
                    score -= 30.0
                if to_square.is_adjacent(game.get_ball_position()):
                    score += 5
                if BotHelper.is_bishop_position_of(game, player, ball_carrier):
                    score -= 2
                score += heat_map.get_cage_necessity_score(to_square)
                if not ball_carrier.state.used:
                    score = max(0.0, score - GrodBot.BASE_SCORE_CAGE_BALL)  # Penalise forming a cage if ball carrier has yet to move
                if not player.state.up:
                    score += 5.0
                return score, True

        return 0, True

    @staticmethod
    def score_mark_opponent(game: g.Game, heat_map: FfHeatMap, player: m.Player, to_square: m.Square) -> Tuple[float, bool]:

        # Modification - no need to mark prone opponents already marked
        ball_carrier = game.get_ball_carrier()
        opp_team = game.get_opp_team(player.team)
        if ball_carrier is not None:
            ball_team = ball_carrier.team
        else:
            ball_team = None
        ball_square = game.get_ball_position()
        if ball_carrier == player:
            return 0.0, True  # Don't mark opponents deliberately with the ball
        all_opponents: List[m.Player] = game.get_adjacent_players(to_square, team=opp_team)
        if not all_opponents:
            return 0.0, True

        if (ball_carrier is not None) and (ball_carrier == player):
            return 0.0, True

        score = GrodBot.BASE_SCORE_MOVE_TO_OPPONENT
        if to_square.is_adjacent(game.get_ball_position()):
            if ball_team == player.team:
                score += 20.0
            else:
                score += 30.0

        for opp in all_opponents:
            if BotHelper.distance_to_scoring_endzone(game, opp.team, to_square) < opp.get_ma() + 2:
                score += 10.0  # Mark opponents in scoring range first.
                break         # Only add score once.

        if len(all_opponents) == 1:
            score += 20.0
            num_friendly_next_to = game.num_tackle_zones_in(all_opponents[0])
            if all_opponents[0].state.up:
                if num_friendly_next_to == 1:
                    score += 5.0
                else:
                    score -= 10.0 * num_friendly_next_to

            if not all_opponents[0].state.up:
                if num_friendly_next_to == 0:
                    score += 5.0
                else:
                    score -= 10.0 * num_friendly_next_to  # Unless we want to start fouling ...

        if not player.state.up:
            score += 25.0
        if not player.has_skill(t.Skill.GUARD):
            score -= len(all_opponents) * 10.0
        else:
            score += len(all_opponents) * 10.0

        if player.get_st() < 3:
            score -= 10

        ball_is_near = False
        for current_opponent in all_opponents:
            if current_opponent.position.is_adjacent(game.get_ball_position()):
                ball_is_near = True

        if ball_is_near:
            score += 8.0
        if player.position != to_square and game.num_tackle_zones_in(player) > 0:
            score -= 40.0

        if ball_square is not None:
            distance_to_ball = ball_square.distance(to_square)
            score -= distance_to_ball / 5.0   # Mark opponents closer to ball when possible

            if ball_team == opp_team:
                if distance_to_ball == 1:
                    score += 15
                elif distance_to_ball == 2:
                    score += 5

        if ball_team is not None and ball_team != player.team:
            distance_ball_carrier_to_end = BotHelper.distance_to_defending_endzone(game, player.team, ball_square)
            distance_square_to_end = BotHelper.distance_to_defending_endzone(game, player.team, to_square)

            if distance_square_to_end + 1.0 < distance_ball_carrier_to_end:
                score += 5.0  # Increase score defending on correct side of field - let's not get caught behind the play

            # if ball_square is not None:
            #     distance_to_ball = ball_square.distance(to_square)
            #     score += 8.0 * max(4.0 - distance_to_ball, 0.0)  # Increase score defending in front of ball carrier
            score -= distance_square_to_end / 10.0  # Increase score a small amount to screen closer to opponents.

            distance_to_other_endzone = BotHelper.distance_to_scoring_endzone(game, player.team, to_square)

            # This way there is a preference for most advanced (distance wise) units.

        return score, True

    @staticmethod
    def score_handoff(game: g.Game, heat_map: FfHeatMap, ball_carrier: m.Player, receiver: m.Player, from_square: m.Square) -> float:
        if receiver == ball_carrier:
            return 0.0

        score = GrodBot.BASE_SCORE_HANDOFF
        score += BotHelper.probability_fail_to_score(BotHelper.probability_catch_fail(game, receiver))
        if not ball_carrier.team.state.reroll_used:
            score += +10.0
        score -= 5.0 * (BotHelper.distance_to_scoring_endzone(game, ball_carrier.team, receiver.position) - BotHelper.distance_to_scoring_endzone(game, ball_carrier.team, ball_carrier.position))
        if receiver.state.used:
            score -= 30.0
        if (game.num_tackle_zones_in(ball_carrier) > 0 or game.num_tackle_zones_in(receiver) > 0) and not BotHelper.blitz_used(game):
            score -= 50.0  # Don't try a risky hand-off if we haven't blitzed yet
        if BotHelper.in_scoring_range(game, receiver) and not BotHelper.in_scoring_range(game, ball_carrier):
            score += 40.0
        # score += heat_map.get_ball_move_square_safety_score(receiver.position)
        return score

    @staticmethod
    def score_pass(game: g.Game, heat_map: FfHeatMap, passer: m.Player, from_square: m.Square, to_square: m.Square) -> float:

        receiver = game.get_player_at(to_square)

        if receiver is None:
            return 0.0
        if receiver.team != passer.team:
            return 0.0
        if receiver == passer:
            return 0.0

        score = GrodBot.BASE_SCORE_PASS
        score += BotHelper.probability_fail_to_score(BotHelper.probability_catch_fail(game, receiver))
        dist: t.PassDistance = game.get_pass_distance(from_square, receiver.position)
        score += BotHelper.probability_fail_to_score(BotHelper.probability_pass_fail(game, passer, from_square, dist))
        if not passer.team.state.reroll_used:
            score += +10.0
        score = score - 5.0 * (BotHelper.distance_to_scoring_endzone(game, receiver.team, receiver.position) - BotHelper.distance_to_scoring_endzone(game, passer.team, passer.position))
        if receiver.state.used:
            score -= 30.0
        if game.num_tackle_zones_in(passer) > 0 or game.num_tackle_zones_in(receiver) > 0 and not BotHelper.blitz_used(game):
            score -= 50.0
        if BotHelper.in_scoring_range(game, receiver) and not BotHelper.in_scoring_range(game, passer):
            score += 40.0
        return score

    @staticmethod
    def score_block(game: g.Game, heat_map: FfHeatMap, attacker: m.Player, defender: m.Player) -> float:
        score = GrodBot.BASE_SCORE_BLOCK
        ball_carrier = game.get_ball_carrier()
        ball_square = game.get_ball_position()
        if attacker.has_skill(t.Skill.CHAINSAW):
            score += 15.0
            score += 20.0 - 2 * defender.get_av()
            # Add something in case the defender is really valuable?
        else:
            num_block_dice = game.num_block_dice(attacker, defender)
            if num_block_dice == 3:
                score += 20.0
            if num_block_dice == 2:
                score += 10.0
            if num_block_dice == 1:
                if attacker.has_skill(t.Skill.BLOCK) or attacker.has_skill(t.Skill.WRESTLE):
                    score -= 36
                else:
                    score += -66.0  # score is close to zero.
            if num_block_dice == -2:
                score += -95.0
            if num_block_dice == -3:
                score += -150.0

            if not attacker.team.state.reroll_used and not attacker.has_skill(t.Skill.LONER):
                score += 10.0
            if attacker.has_skill(t.Skill.BLOCK) or attacker.has_skill(t.Skill.WRESTLE):
                score += 20.0
            if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE):
                score += -10.0
            if defender.has_skill(t.Skill.BLOCK):
                score += -10.0
            if BotHelper.attacker_would_surf(game, attacker, defender):
                score += 32.0
            if attacker.has_skill(t.Skill.LONER):
                score -= 10.0

        if attacker == ball_carrier:
            score += -45.0
        if defender == ball_carrier:
            score += 40.0
        if (ball_square is not None) and defender.position.is_adjacent(ball_square):
            score += 15.0

        return score

    @staticmethod
    def score_push(game: g.Game, from_square: m.Square, to_square: m.Square) -> float:
        score = 0.0
        ball_square = game.get_ball_position()
        if BotHelper.distance_to_sideline(game, to_square) == 0:
            score = score + 10.0    # Push towards sideline
        if ball_square is not None and to_square.is_adjacent(ball_square):
            score = score - 15.0    # Push away from ball
        if BotHelper.direct_surf_squares(game, from_square, to_square):
            score = score + 10.0
        return score

    @staticmethod
    def check_follow_up(game: g.Game) -> bool:
        # To do: the  logic here is faulty for the current game state,  in terms of how and when actions are evaluated.  I.e.
        # the check appears to happen before the defending player is placed prone (but after the player is pushed?)
        # What I want is to follow up, generally, if the defender is prone and not otherwise.
        active_player: m.Player = game.state.active_player

        block_proc = BotHelper.last_block_proc(game)

        attacker: m.Player = block_proc.attacker
        defender: m.Player = block_proc.defender
        is_blitz_action = block_proc.blitz
        for position in game.state.available_actions[0].positions:
            if active_player.position != position:
                follow_up_square: m.Square = position

        defender_prone = (block_proc.selected_die == t.BBDieResult.DEFENDER_DOWN) or ((block_proc.selected_die == t.BBDieResult.DEFENDER_STUMBLES) and (attacker.has_skill(t.Skill.TACKLE) or not defender.has_skill(t.Skill.DODGE)))

        num_tz_cur = game.num_tackle_zones_in(active_player)
        num_tz_new = game.num_tackle_zones_at(active_player, follow_up_square)
        opp_adj_cur = game.get_adjacent_opponents(active_player, stunned=False, down=False)
        opp_adj_new = game.get_adjacent_players(follow_up_square, team=game.get_opp_team(active_player.team), stunned=False, down=False)

        num_tz_new -= defender_prone

        # If blitzing (with squares of movement left) always follow up if the new square is not in any tackle zone.
        if is_blitz_action and attacker.num_moves_left() > 0 and num_tz_new == 0:
            return True

        # If Attacker has the ball, strictly follow up only if there are less opponents next to new square.
        if game.get_ball_carrier == attacker:
            # This isn't working
            return False
            if len(opp_adj_new) - defender_prone < len(opp_adj_cur) - 1:
                return True
            else:
                return False

        if game.get_ball_carrier == defender:
            return True   # Always follow up if defender has ball
        if BotHelper.distance_to_sideline(game, follow_up_square) == 0:
            return False    # No if moving to sideline
        if BotHelper.distance_to_sideline(game, defender.position) == 0:
            return True  # Follow up if opponent is on sideline
        if follow_up_square.is_adjacent(game.get_ball_position()):
            return True  # Follow if moving next to ball
        if attacker.position.is_adjacent(game.get_ball_position()):
            return False  # Don't follow if already next to ball

        # Follow up if less standing opponents in the next square or equivalent, but defender is now prone
        if (num_tz_new == 0) or (num_tz_new < num_tz_cur) or (num_tz_new == num_tz_cur and not defender_prone):
            return True
        if attacker.has_skill(t.Skill.GUARD) and num_tz_new > num_tz_cur:
            return True      # Yes if attacker has guard
        if attacker.get_st() > defender.get_st() + num_tz_new - num_tz_cur:
            return True  # Follow if stronger
        if is_blitz_action and attacker.num_moves_left() == 0:
            return True  # If blitzing but out of moves, follow up to prevent GFIing...

        return False

    @staticmethod
    def check_reroll_block(game: g.Game, team: m.Team, block_results: List[ActionSequence], favor: m.Team) -> bool:
        block_proc: Optional[p.Block] = BotHelper.last_block_proc(game)
        attacker: m.Player = block_proc.attacker
        defender: m.Player = block_proc.defender
        is_blitz_action = block_proc.blitz
        ball_carrier: Optional[m.Player] = game.get_ball_carrier()

        best_block_score: float = 0
        cur_block_score: float = -1

        if len(block_results) > 0:
            best_block_score = block_results[0].score

        if len(block_results) > 1:
            cur_block_score = block_results[1].score
            if favor == team and cur_block_score > best_block_score:
                best_block_score = cur_block_score
            if favor != team and cur_block_score < best_block_score:
                best_block_score = cur_block_score

        if len(block_results) > 2:
            cur_block_score = block_results[2].score
            if favor == team and cur_block_score > best_block_score:
                best_block_score = cur_block_score
            if favor != team and cur_block_score < best_block_score:
                best_block_score = cur_block_score

        if best_block_score < 3:
            return True
        elif ball_carrier == defender and best_block_score < 5:
            return True  # Reroll if target has ball and not knocked over.
        else:
            return False

    @staticmethod
    def scoring_urgency_score(game: g.Game, heat_map: FfHeatMap, player: m.Player) -> float:
        if player.team.state.turn == 8:
            return 40
        return 0

    @staticmethod
    def path_cost_to_score(path: pf.Path) -> float:
        score = -((1-path.prob) * (1-path.prob) * (250.0 + GrodBot.ADDITIONAL_SCORE_DODGE))

        if path.prob < 1:
            score -= 10

        return score

    @staticmethod
    def continuation_path_cost_to_score(path: pf.Path) -> float:
        score = - (1-path.prob) * (1-path.prob) * (250.0 + GrodBot.ADDITIONAL_SCORE_DODGE) * 4
        return score


    @staticmethod
    def probability_fail_to_score(probability: float) -> float:
        score = -(probability * probability * (250.0 + GrodBot.ADDITIONAL_SCORE_DODGE))
        return score

    @staticmethod
    def probability_catch_fail(game: g.Game, receiver: m.Player) -> float:
        num_tz = 0.0
        if not receiver.has_skill(t.Skill.NERVES_OF_STEEL):
            num_tz = game.num_tackle_zones_in(receiver)
        probability_success = min(5.0, receiver.get_ag()+1.0-num_tz)/6.0
        if receiver.has_skill(t.Skill.CATCH):
            probability_success += (1.0-probability_success)*probability_success
        probability = 1.0 - probability_success
        return probability

    @staticmethod
    def probability_pass_fail(game: g.Game, passer: m.Player, from_square: m.Square, dist: t.PassDistance) -> float:
        num_tz = 0.0
        if not passer.has_skill(t.Skill.NERVES_OF_STEEL):
            num_tz = game.num_tackle_zones_at(passer, from_square)
        if passer.has_skill(t.Skill.ACCURATE):
            num_tz -= 1
        if passer.has_skill(t.Skill.STRONG_ARM and dist != t.PassDistance.QUICK_PASS):
            num_tz -= 1
        if dist == t.PassDistance.HAIL_MARY:
            return -100.0
        if dist == t.PassDistance.QUICK_PASS:
            num_tz -= 1
        if dist == t.PassDistance.SHORT_PASS:
            num_tz -= 0
        if dist == t.PassDistance.LONG_PASS:
            num_tz += 1
        if dist == t.PassDistance.LONG_BOMB:
            num_tz += 2
        probability_success = min(5.0, passer.get_ag()-num_tz)/6.0
        if passer.has_skill(t.Skill.PASS):
            probability_success += (1.0-probability_success)*probability_success
        probability = 1.0 - probability_success
        return probability

    @staticmethod
    def choose_gaze_victim(game: g.Game, player: m.Player) -> m.Player:
        best_victim: Optional[m.Player] = None
        best_score = 0.0
        ball_square: m.Square = game.get_ball_position()
        potentials: List[m.Player] = game.get_adjacent_players(player, team=game.get_opp_team(player.team), down=False, standing=True, stunned=False)
        for unit in potentials:
            current_score = 5.0
            current_score += 6.0 - unit.get_ag()
            if (ball_square is not None) and unit.position.is_adjacent(ball_square):
                current_score += 5.0
            if current_score > best_score:
                best_score = current_score
                best_victim = unit
        return best_victim

    @staticmethod
    def average_st(game: g.Game, players: List[m.Player]) -> float:
        values = [player.get_st() for player in players]
        return sum(values)*1.0 / len(values)

    @staticmethod
    def average_av(game: g.Game, players: List[m.Player]) -> float:
        values = [player.get_av() for player in players]
        return sum(values)*1.0 / len(values)

    @staticmethod
    def average_ma(game: g.Game, players: List[m.Player]) -> float:
        values = [player.get_ma() for player in players]
        return sum(values)*1.0 / len(values)

    @staticmethod
    def player_bash_ability(game: g.Game, player: m.Player) -> float:
        bashiness: float = 0.0
        bashiness += 10.0 * player.get_st()
        bashiness += 5.0 * player.get_av()
        if player.has_skill(t.Skill.BLOCK):
            bashiness += 10.0
        if player.has_skill(t.Skill.WRESTLE):
            bashiness += 10.0
        if player.has_skill(t.Skill.MIGHTY_BLOW):
            bashiness += 5.0
        if player.has_skill(t.Skill.CLAWS):
            bashiness += 5.0
        if player.has_skill(t.Skill.PILING_ON):
            bashiness += 5.0
        if player.has_skill(t.Skill.GUARD):
            bashiness += 15.0
        if player.has_skill(t.Skill.DAUNTLESS):
            bashiness += 10.0
        if player.has_skill(t.Skill.FOUL_APPEARANCE):
            bashiness += 5.0
        if player.has_skill(t.Skill.TENTACLES):
            bashiness += 5.0
        if player.has_skill(t.Skill.STUNTY):
            bashiness -= 10.0
        if player.has_skill(t.Skill.REGENERATION):
            bashiness += 10.0
        if player.has_skill(t.Skill.THICK_SKULL):
            bashiness += 3.0
        if player.has_skill(t.Skill.PASS):
            bashiness -= 5.0
        if player.has_skill(t.Skill.SURE_HANDS):
            bashiness -= 5.0
        return bashiness

    @staticmethod
    def team_bash_ability(game: g.Game, players: List[m.Player]) -> float:
        total = 0.0
        for player in players:
            total += BotHelper.player_bash_ability(game, player)
        return total

    @staticmethod
    def player_pass_ability(game: g.Game, player: m.Player) -> float:
        passing_ability = 0.0
        passing_ability += player.get_ag() * 15.0    # Agility most important.
        passing_ability += player.get_ma() * 2.0     # Fast movements make better ball throwers.
        if player.has_skill(t.Skill.PASS):
            passing_ability += 10.0
        if player.has_skill(t.Skill.SURE_HANDS):
            passing_ability += 5.0
        if player.has_skill(t.Skill.EXTRA_ARMS):
            passing_ability += 3.0
        if player.has_skill(t.Skill.NERVES_OF_STEEL):
            passing_ability += 3.0
        if player.has_skill(t.Skill.ACCURATE):
            passing_ability += 5.0
        if player.has_skill(t.Skill.STRONG_ARM):
            passing_ability += 5.0
        if player.has_skill(t.Skill.BONE_HEAD):
            passing_ability -= 15.0
        if player.has_skill(t.Skill.REALLY_STUPID):
            passing_ability -= 15.0
        if player.has_skill(t.Skill.WILD_ANIMAL):
            passing_ability -= 15.0
        if player.has_skill(t.Skill.ANIMOSITY):
            passing_ability -= 10.0
        if player.has_skill(t.Skill.LONER):
            passing_ability -= 15.0
        if player.has_skill(t.Skill.DUMP_OFF):
            passing_ability += 5.0
        if player.has_skill(t.Skill.SAFE_THROW):
            passing_ability += 5.0
        if player.has_skill(t.Skill.NO_HANDS):
            passing_ability -= 100.0
        return passing_ability

    @staticmethod
    def player_blitz_ability(game: g.Game, player: m.Player) -> float:
        blitzing_ability = BotHelper.player_bash_ability(game, player)
        blitzing_ability += player.get_ma() * 10.0
        if player.has_skill(t.Skill.TACKLE):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.SPRINT):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.SURE_FEET):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.STRIP_BALL):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.DIVING_TACKLE):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.MIGHTY_BLOW):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.CLAWS):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.PILING_ON):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.BONE_HEAD):
            blitzing_ability -= 15.0
        if player.has_skill(t.Skill.REALLY_STUPID):
            blitzing_ability -= 15.0
        if player.has_skill(t.Skill.WILD_ANIMAL):
            blitzing_ability -= 10.0
        if player.has_skill(t.Skill.LONER):
            blitzing_ability -= 15.0
        if player.has_skill(t.Skill.SIDE_STEP):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.JUMP_UP):
            blitzing_ability += 5.0
        if player.has_skill(t.Skill.HORNS):
            blitzing_ability += 10.0
        if player.has_skill(t.Skill.JUGGERNAUT):
            blitzing_ability += 10.0
        if player.has_skill(t.Skill.LEAP):
            blitzing_ability += 5.0
        return blitzing_ability

    @staticmethod
    def player_receiver_ability(game: g.Game, player: m.Player) -> float:
        receiving_ability = 0.0
        receiving_ability += player.get_ma() * 5.0
        receiving_ability += player.get_ag() * 10.0
        if player.has_skill(t.Skill.CATCH):
            receiving_ability += 15.0
        if player.has_skill(t.Skill.EXTRA_ARMS):
            receiving_ability += 10.0
        if player.has_skill(t.Skill.NERVES_OF_STEEL):
            receiving_ability += 5.0
        if player.has_skill(t.Skill.DIVING_CATCH):
            receiving_ability += 5.0
        if player.has_skill(t.Skill.DODGE):
            receiving_ability += 10.0
        if player.has_skill(t.Skill.SIDE_STEP):
            receiving_ability += 5.0
        if player.has_skill(t.Skill.BONE_HEAD):
            receiving_ability -= 15.0
        if player.has_skill(t.Skill.REALLY_STUPID):
            receiving_ability -= 15.0
        if player.has_skill(t.Skill.WILD_ANIMAL):
            receiving_ability -= 15.0
        if player.has_skill(t.Skill.LONER):
            receiving_ability -= 15.0
        if player.has_skill(t.Skill.NO_HANDS):
            receiving_ability -= 100.0
        return receiving_ability

    @staticmethod
    def player_run_ability(game: g.Game, player: m.Player) -> float:
        running_ability = 0.0
        running_ability += player.get_ma() * 10.0    # Really favour fast units
        running_ability += player.get_ag() * 10.0    # Agility to be prized
        running_ability += player.get_st() * 5.0     # Doesn't hurt to be strong!
        if player.has_skill(t.Skill.SURE_HANDS):
            running_ability += 10.0
        if player.has_skill(t.Skill.BLOCK):
            running_ability += 10.0
        if player.has_skill(t.Skill.EXTRA_ARMS):
            running_ability += 5.0
        if player.has_skill(t.Skill.DODGE):
            running_ability += 10.0
        if player.has_skill(t.Skill.SIDE_STEP):
            running_ability += 5.0
        if player.has_skill(t.Skill.STAND_FIRM):
            running_ability += 3.0
        if player.has_skill(t.Skill.BONE_HEAD):
            running_ability -= 15.0
        if player.has_skill(t.Skill.REALLY_STUPID):
            running_ability -= 15.0
        if player.has_skill(t.Skill.WILD_ANIMAL):
            running_ability -= 15.0
        if player.has_skill(t.Skill.LONER):
            running_ability -= 15.0
        if player.has_skill(t.Skill.ANIMOSITY):
            running_ability -= 5.0
        if player.has_skill(t.Skill.DUMP_OFF):
            running_ability += 5.0
        if player.has_skill(t.Skill.NO_HANDS):
            running_ability -= 100.0
        return running_ability

    @staticmethod
    def player_value(game: g.Game, player: m.Player) -> float:
        value = player.get_ag()*40 + player.get_av()*30 + player.get_ma()*30 + player.get_st()*50 + len(player.get_skills())*20
        return value

    @staticmethod
    def is_do_nothing(action):
        type_1 = (len(action.action_steps) == 2) and (action.action_steps[0].action_type == t.ActionType.START_MOVE) and (action.action_steps[1].action_type == t.ActionType.END_PLAYER_TURN)
        type_2 = (len(action.action_steps) == 1) and (action.action_steps[0].action_type == t.ActionType.START_MOVE)
        return type_1 or type_2


# Register bot
botbowl.register_bot('GrodBot', GrodBot)

def main():
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = True
    # config = get_config("gym-7.json")
    # config = get_config("gym-5.json")
    # config = get_config("gym-3.json")
    ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)

    num_games = 10
    wins = 0
    tds = 0
    # Play 10 games
    for i in range(num_games):
        home_agent = botbowl.make_bot('Grodbot')
        home_agent.name = "GrodBot"
        away_agent = botbowl.make_bot('random')
        away_agent.name = "Random Bot"
        config.debug_mode = False
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)

        wins += 1 if game.get_winning_team() is game.state.home_team else 0
        tds += game.state.home_team.state.score
    print(f"won {wins}/{num_games}")
    print(f"Own TDs per game={tds/num_games}")

    
if __name__ == "__main__":
    main()
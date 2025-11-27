from SkullGame import SkullGame
from RandomPlayer import RandomPlayer
from DocilePlayer import DocilePlayer
from PacifistPlayer import PacifistPlayer
from AggressivePlayer import AggressivePlayer
from ObservantPlayer import ObservantPlayer
from EarlyBidder import EarlyBidder
import statistics

def simulate(num_games=200, verbose=False):
    results = {
        "Aggressive": 0,
        "Pacifist": 0,
        "Docile": 0,
        "Random": 0,
        "EarlyBidder" : 0
    }

    eliminations = {
        "Aggressive": 0,
        "Pacifist": 0,
        "Docile": 0,
        "Random": 0,
        "EarlyBidder" : 0
    }

    cards_remaining = {
        "Aggressive": [],
        "Pacifist": [],
        "Docile": [],
        "Random": [],
        "EarlyBidder" : []
    }

    elimination_wins = 0
    point_wins = 0
    round_counts = []

    for game_index in range(num_games):
        players = [
            AggressivePlayer("AggressiveAndy"),
            PacifistPlayer("PacifistPete"),
            DocilePlayer("DocileDora"),
            RandomPlayer("RandomRandy"),
            EarlyBidder("EarlyEddie")
        ]

        game = SkullGame(players)
        rounds_played = 0

        if verbose:
            print(f"\n=== Starting Game {game_index + 1} ===")

        # --- Run game manually to count rounds ---
        while (
            len([p for p in game.players if p.active and p.hand]) > 1
            and not any(p.score >= 2 for p in game.players)
        ):
            game.play_round()
            rounds_played += 1

        round_counts.append(rounds_played)

        # --- Determine the winner ---
        active_players = [p for p in game.players if p.active and p.hand]
        if len(active_players) == 1:
            winner = active_players[0]
            elimination_wins += 1
        else:
            winner = max(game.players, key=lambda p: p.score)
            point_wins += 1

        # --- Record the winner type ---
        if isinstance(winner, AggressivePlayer):
            results["Aggressive"] += 1
        elif isinstance(winner, PacifistPlayer):
            results["Pacifist"] += 1
        elif isinstance(winner, DocilePlayer):
            results["Docile"] += 1
        elif isinstance(winner, RandomPlayer):
            results["Random"] += 1
        elif isinstance(winner, EarlyBidder):
            results["EarlyBidder"] += 1

        # --- Record elimination and hand stats ---
        for p in players:
            player_type = p.__class__.__name__.replace("Player", "")
            if not p.active or not p.hand:
                eliminations[player_type] += 1
            cards_remaining[player_type].append(len(p.hand))

        # --- Per-game end summary ---
        if verbose:
            print(f"\nGame {game_index + 1} Summary:")
            print(f"Winner: {winner.name} ({winner.__class__.__name__})")
            print(f"Rounds: {rounds_played}")
            print("Final Player States:")
            for p in players:
                status = "ELIMINATED ❌" if not p.active or not p.hand else "ACTIVE ✅"
                print(f"  {p.name:15} | Cards Left: {len(p.hand)} | Score: {p.score} | {status}")
            print("-" * 40)

        if (game_index + 1) % 50 == 0 and not verbose:
            print(f"Completed {game_index + 1}/{num_games} games...")

    # --- Summary statistics ---
    print("\n=== Simulation Results ===")
    total = sum(results.values())
    for k, v in results.items():
        print(f"{k:10}: {v:4} wins ({v/total:.1%})")

    print("\n=== Win Type Breakdown ===")
    print(f"Elimination Wins: {elimination_wins} ({elimination_wins / num_games:.1%})")
    print(f"Point Wins:       {point_wins} ({point_wins / num_games:.1%})")

    print("\n=== Game Length Statistics ===")
    if round_counts:
        print(f"Average Rounds per Game: {statistics.mean(round_counts):.2f}")
        print(f"Median Rounds per Game:  {statistics.median(round_counts):.2f}")
        print(f"Shortest Game: {min(round_counts)} rounds")
        print(f"Longest Game:  {max(round_counts)} rounds")

    print("\n=== Player Survival & Card Statistics ===")
    for ptype in results.keys():
        avg_cards = statistics.mean(cards_remaining[ptype]) if cards_remaining[ptype] else 0
        print(f"{ptype:10} | Eliminated: {eliminations[ptype]:4}/{num_games} | "
              f"Avg. Cards Left: {avg_cards:.2f}")

    return {
        "wins": results,
        "eliminations": eliminations,
        "cards_remaining": cards_remaining,
        "elimination_wins": elimination_wins,
        "point_wins": point_wins,
        "round_counts": round_counts
    }

if __name__ == "__main__":
    simulate(100000, verbose=False)

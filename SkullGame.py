import random
from RandomPlayer import RandomPlayer
from DocilePlayer import DocilePlayer
from PacifistPlayer import PacifistPlayer
from AggressivePlayer import AggressivePlayer
from ObservantPlayer import ObservantPlayer
from EarlyBidder import EarlyBidder

class SkullGame:
    def __init__(self, players):
        self.players = players 
        self.current_bid = 0
        self.bid_winner = None
        self.active_bidders = []

    def play_round(self):
        # Include only active players at the start of each round; thus allowing for elimination
        active_players = [p for p in self.players if p.active and p.hand]
        if len(active_players) <= 1:
            # Not enough players to continue end game
            return

        # 🎲 Randomize who starts each round
        start_index = random.randint(0, len(active_players) - 1)
        active_players = active_players[start_index:] + active_players[:start_index]

        print("\n=== NEW ROUND ===")
        print(f"Active players (starting from {active_players[0].name}): {[p.name for p in active_players]}")

        # Show each player's current hand at round start for testing/bug fixing reasons
        for p in active_players:
            hand_summary = ", ".join(card.kind for card in p.hand)
            print(f"  {p.name}'s hand ({len(p.hand)} cards): [{hand_summary}]")

        # Phase 1: Placement
        self.placement_phase(active_players)

        # Phase 2: Bidding (triggered during Placement Phase function)
        if not any(p.pile for p in active_players):
            print("No cards on table, skipping round.")
            return

        self.bidding_phase(active_players)

        # Phase 3: Reveal
        if self.bid_winner:
            self.reveal_phase(self.bid_winner)


    def placement_phase(self, active_players):
        print("\n--- PLACEMENT ---")
        bidding_triggered = False
        turn = 0

        while not bidding_triggered:
            current_player = active_players[turn % len(active_players)]

            # Random decision: 70% place, 30% bid
            decision = current_player.time_to_bid()

            # Won't let bidding start if there are no cards in the pile.
            if decision == "bid" and (len(p.pile) >= 1 for p in active_players):
                decision = "place"

            if decision == "place" and current_player.hand:
                current_player.play_card()
                print(f"{current_player.name} places a card. Total in pile: {len(current_player.pile)}")
            else:
                print(f"{current_player.name} starts the bidding!")
                bidding_triggered = True
                self.starting_player = current_player
                break

            turn += 1


    def bidding_phase(self, active_players):
        print("\n--- BIDDING ---")
        self.active_bidders = [p for p in active_players if p.active and p.pile]
        self.current_bid = 0
        folded = set()
        self.bid_winner = None

        # Start with bidding initiator
        current_index = active_players.index(self.starting_player)
        current_player = self.starting_player

        # Starting player bid using logic to determine the maximum possible bid size (according to the number of cards in the collective pile)
        max_possible = sum(len(p.pile) for p in active_players)
        self.current_bid = int(current_player.choose_bid(0, max_possible) or 1)
        self.bid_winner = current_player
        print(f"{current_player.name} opens the bidding with {self.current_bid}!")

        current_index += 1

        # Continue bidding process
        while len(self.active_bidders) - len(folded) > 1 and self.current_bid < max_possible:
            current_player = active_players[current_index % len(active_players)]
            if current_player not in self.active_bidders or current_player in folded:
                current_index += 1
                continue

            new_bid = current_player.choose_bid(self.current_bid, max_possible)

            if new_bid is None:
                print(f"{current_player.name} folds.")
                folded.add(current_player)
            else:
                self.current_bid = new_bid
                self.bid_winner = current_player
                print(f"{current_player.name} raises to {self.current_bid}")

            current_index += 1

        # Should never occur but bug fixing logic for logical errors where somehow all players can fold.
        if self.bid_winner is None:
            print("Everyone folded! No one wins the bid this round.")
            return

        print(f"{self.bid_winner.name} wins the bidding with {self.current_bid}!")

    def reveal_phase(self, player):
        print("\n--- REVEAL ---")
        revealed = []
        to_reveal = self.current_bid

        # Reveal own cards first
        while player.pile and len(revealed) < to_reveal:
            card = player.pile.pop()
            revealed.append((player,card))
            print(f"{player.name} reveals their own {card.kind}")
            if card.kind == "skull":
                print(f"{player.name} revealed a skull!")
                player.lose_random_card()
                self.reset_round_hands()
                return

        # Use player logic for card reveal
        success = player.reveal_cards(self, to_reveal, revealed)
        if not success:
            print(f"{player.name} revealed a skull!")
            player.lose_random_card()
            self.reset_round_hands()
            return

        # Success — all reveals safe
        print(f"{player.name} successfully revealed {len(revealed)} cards!")
        player.score += 1
        self.reset_round_hands()

    def reset_round_hands(self):
        for p in self.players:
            p.pile.clear()

            # Build the restored hand
            p.hand = p.default_hand.copy()

        # Debug info
        for p in self.players:
            print(f"{p.name} resets to {len(p.hand)} cards.")

    def play_game(self):
        while (
            len([p for p in self.players if p.active and p.hand]) > 1
            and not any(p.score >= 2 for p in self.players)
        ):
            self.play_round()

        # Determine the winner
        active_players = [p for p in self.players if p.active and p.hand]
        if len(active_players) == 1:
            winner = active_players[0]
            print(f"\n{winner.name} wins by elimination!")
        else:
            winner = max(self.players, key=lambda p: p.score)
            print(f"\n{winner.name} wins with {winner.score} points!")

        # --- Post-game summary ---

        print("\n=== FINAL RESULTS ===")

        for p in self.players:
            #Asked chatGPT how to make this more readable in the VSCode terminal hence the emojis. It is the only section where
            #AI was used in this code file.
            status = "ELIMINATED ❌" if not p.active or not p.hand else "ACTIVE ✅"
            print(f"{p.name:10} | Score: {p.score:<2} | Cards Left: {len(p.hand):<1} | {status}")


if __name__ == "__main__":
    # Example players
    players = [
        RandomPlayer("RandomRandy"),
        RandomPlayer("RandomRuth"),
        AggressivePlayer("AggressiveAndy"),
        PacifistPlayer("PacifistPete"),
        PacifistPlayer("PacifistPeter"),
        PacifistPlayer("PacifistPaul"),
        DocilePlayer("DocileDora"),
        ObservantPlayer("ObservantOdie"),
        EarlyBidder("EarlyErnest")
    ]

    # Run the game
    game = SkullGame(players)
    game.play_game()

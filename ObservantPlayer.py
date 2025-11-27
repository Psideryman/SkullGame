import random
from Card import Card
from Player import Player

class ObservantPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        # Track observed skull positions per player
        self.skull_positions = {}  # {player_name: [positions]}
        self.flower_positions = {} # {player_name: [positions]}

    def play_card(self):
        """
        Place flowers first, skulls later to avoid self-elimination.
        """
        flowers = [c for c in self.hand if c.kind == "flower"]
        skulls = [c for c in self.hand if c.kind == "skull"]

        if flowers:
            card = flowers[0]
        else:
            card = skulls[0]

        self.hand.remove(card)
        self.pile.append(card)

    def choose_bid(self, current_bid, max_possible):
        """
        Decide the bid based on observed average skull positions:
        - Estimate safe draws for each player (cards before avg skull position).
        - Determine maximum safe bid so all own flowers can be drawn safely.
        """
        safe_draws = 0
        for p_name, positions in self.skull_positions.items():
            if positions:
                avg_skull_pos = sum(positions) / len(positions)
                # Only count the cards before the skull as safe
                safe_draws += int(avg_skull_pos - 1)
            else:
                # If we have no info, assume all cards are safe
                safe_draws += max(len(self.pile), 0)

        # Add own flowers
        own_flowers = len([c for c in self.hand if c.kind == "flower"])
        safe_draws += own_flowers

        # Bid only if safe_draws exceeds current bid
        if safe_draws > current_bid:
            # Increase bid slightly, but not more than safe_draws or max_possible
            bid_increase = random.randint(1, min(2, safe_draws - current_bid))
            return min(current_bid + bid_increase, max_possible)
        return None  # fold if risky

    def reveal_cards(self, game, to_reveal, revealed):
        """
        Reveal cards from other players and update skull/flower positions.
        """
        others = [p for p in game.players if p != self and p.active and p.pile]

        while len(revealed) < to_reveal and others:
            target = random.choice(others)

            card = target.pile.pop()
            revealed.append((target, card))

            # Initialize dictionaries
            if target.name not in self.skull_positions:
                self.skull_positions[target.name] = []
            if target.name not in self.flower_positions:
                self.flower_positions[target.name] = []

            # Record card position (from bottom of pile)
            position = len(target.pile) + 1
            if card.kind == "skull":
                self.skull_positions[target.name].append(position)
            else:
                self.flower_positions[target.name].append(position)

            if card.kind == "skull":
                return False  # skull ends the round
            if not target.pile:
                others.remove(target)

        return True, revealed
    
    def time_to_bid(self):
        """
        Place cards first until the pile is sizable, then bid strategically.
        """
        if len(self.pile) < 2:
            return "place"
        return "bid"

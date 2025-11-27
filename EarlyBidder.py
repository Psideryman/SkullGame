import random
from Card import Card
from Player import Player

class EarlyBidder(Player):
    def __init__(self, name):
        super().__init__(name)

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
        return min(max_possible, current_bid+1)

    def reveal_cards(self, game, to_reveal, revealed):
        # Reveal cards from other players until a skull is found
        others = [p for p in game.players if p != self and p.active and p.pile]

        while len(revealed) < to_reveal and others:
            target = random.choice(others)
            card = target.pile.pop()
            revealed.append((target, card))
            print(f"{self.name} reveals {target.name}'s {card.kind}")

            if card.kind == "skull":
                return False  # round ends if skull is revealed

            if not target.pile:
                others.remove(target)

        return True, revealed
    
    def time_to_bid(self):
        """
        Place cards first until the pile is sizable, then bid strategically.
        """
        return "bid"

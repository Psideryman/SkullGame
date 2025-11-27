from Player import Player
import random

class AggressivePlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.hasBid = False

    def play_card(self):
        self.hasBid = False
        if not self.hand:
            self.active = False
            return
        
        flowers = [c for c in self.hand if c.kind == "flower"]
        skulls = [c for c in self.hand if c.kind == "skull"]
        
        # Play skull first, then flower on 2nd or 3rd placement
        if len(self.pile) == 0:
            # First card: play skull
            card = flowers[0] if flowers else skulls[0]
        elif len(self.pile) in [1, 2]:
            # 2nd or 3rd card: play flower
            card = skulls[0] if skulls else flowers[0]
        else:
            # After that: play whatever is left
            if flowers:
                card = flowers[0]
            elif skulls:
                card = skulls[0]
            else:
                card = random.choice(self.hand)
        
        self.hand.remove(card)
        self.pile.append(card)

    def time_to_bid(self):
        # Aggressive: loves bidding early to control the round.
        base_chance = 0.6
        if len(self.pile) >= 2:
            base_chance += 0.2  # even more eager if they’ve got bait down 
        return "bid" if random.random() < base_chance else "place"


    def choose_bid(self, current_bid, max_possible):
        # Aggressive: tries to bid high or if it's bid before
        if current_bid >= max_possible or self.hasBid is True:
            print("!!!!!!!!!!!")
            return None
        # Randomly raise 1-3 above current bid
        self.hasBid = True
        return min(max_possible, current_bid + random.randint(1, max(1,max_possible//2)))

    def reveal_cards(self, game, to_reveal, revealed):
        # Similar to RandomPlayer
        others = [p for p in game.players if p != self and p.active and p.pile]

        while len(revealed) < to_reveal and others:
            target = random.choice(others)
            card = target.pile.pop()
            revealed.append((target, card))
            print(f"{self.name} reveals {target.name}'s {card.kind}")
            if card.kind == "skull":
                return False
            if not target.pile:
                others.remove(target)
        return True, revealed

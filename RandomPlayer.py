import random
from Player import Player

class RandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.current_strategy = None
        self.hasBid = False  # For aggressive strategy
        self.last_pile_size = 0  # Track pile size to detect new rounds
        
    def _detect_new_round(self):
        """Detect if a new round has started and pick a new strategy"""
        # New round detected when pile is cleared (goes from >0 to 0)
        if len(self.pile) == 0 and self.last_pile_size > 0:
            self.current_strategy = random.choice(['aggressive', 'pacifist', 'docile'])
            print(f"{self.name} adopting {self.current_strategy} strategy for this round")
        # First round of the game
        elif self.current_strategy is None:
            self.current_strategy = random.choice(['aggressive', 'pacifist', 'docile'])
            print(f"{self.name} adopting {self.current_strategy} strategy for this round")
        
        self.last_pile_size = len(self.pile)

    def play_card(self):
        self._detect_new_round()  # Check for new round at start of action
        
        if not self.hand:
            self.active = False
            return
        
        self.hasBid = False  # Reset for aggressive strategy
        
        flowers = [c for c in self.hand if c.kind == "flower"]
        skulls = [c for c in self.hand if c.kind == "skull"]
        
        if self.current_strategy == 'aggressive':
            # Aggressive: flowers first, then skulls
            if flowers:
                card = flowers[0]
            elif skulls:
                card = skulls[0]
            else:
                card = random.choice(self.hand)
                
        elif self.current_strategy == 'pacifist':
            # Pacifist: skulls first, then flowers
            card = skulls[0] if skulls else flowers[0]
            
        elif self.current_strategy == 'docile':
            # Docile: skulls first, then flowers
            card = skulls[0] if skulls else flowers[0]
        
        else:
            # Fallback (shouldn't happen)
            card = random.choice(self.hand)
        
        self.hand.remove(card)
        self.pile.append(card)

    def time_to_bid(self):
        if self.current_strategy == 'aggressive':
            # Aggressive: loves bidding early (60% base, +20% if pile >= 2)
            base_chance = 0.6
            if len(self.pile) >= 2:
                base_chance += 0.2
            return "bid" if random.random() < base_chance else "place"
            
        elif self.current_strategy == 'pacifist':
            # Pacifist: avoids bidding (10% chance only after 3+ cards)
            if len(self.pile) >= 3 and random.random() < 0.1:
                return "bid"
            return "place"
            
        elif self.current_strategy == 'docile':
            # Docile: prefers to place (20% chance only after 2+ cards)
            if len(self.pile) >= 2 and random.random() < 0.2:
                return "bid"
            return "place"
        
        return "place"

    def choose_bid(self, current_bid, max_possible):
        if self.current_strategy == 'aggressive':
            # Aggressive: tries to bid high, raises by 1-3
            if current_bid >= max_possible or self.hasBid is True:
                return None
            self.hasBid = True
            return min(max_possible, current_bid + random.randint(1, 3))
            
        elif self.current_strategy == 'pacifist':
            # Pacifist: conservative, max bid is pile_size - 1
            max_bid = max(0, len(self.pile) - 1)
            if current_bid >= max_bid:
                return None
            return current_bid + 1
            
        elif self.current_strategy == 'docile':
            # Docile: conservative, max bid is pile_size - 1
            max_bid = max(0, len(self.pile) - 1)
            if current_bid >= max_bid:
                return None
            return current_bid + 1
        
        return None

    def reveal_cards(self, game, to_reveal, revealed):
        if self.current_strategy == 'aggressive':
            # Aggressive: reveal others first (risky)
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
            
        elif self.current_strategy in ['pacifist', 'docile']:
            # Pacifist/Docile: reveal own cards first (safe)
            while self.pile and len(revealed) < to_reveal:
                card = self.pile.pop()
                revealed.append((self, card))
                print(f"{self.name} reveals their own {card.kind}")
                if card.kind == "skull":
                    return False
            
            # Then reveal others
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
        
        return True, revealed
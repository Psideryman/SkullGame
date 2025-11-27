import random
from SkullGame import SkullGame
from RandomPlayer import RandomPlayer
from DocilePlayer import DocilePlayer
from PacifistPlayer import PacifistPlayer
from AggressivePlayer import AggressivePlayer
from EarlyBidder import EarlyBidder

class EvolutionaryAgent:
    """Wrapper for an agent that can change strategies over time"""
    def __init__(self, agent_id, initial_strategy):
        self.agent_id = agent_id
        self.strategy_type = initial_strategy
        self.total_wins = 0
        self.games_played = 0
        self.strategy_history = [initial_strategy]
        
    def create_player_instance(self):
        """Create a new player instance based on current strategy"""
        name = f"Agent{self.agent_id}"
        if self.strategy_type == "Random":
            return RandomPlayer(name)
        elif self.strategy_type == "Aggressive":
            return AggressivePlayer(name)
        elif self.strategy_type == "Pacifist":
            return PacifistPlayer(name)
        elif self.strategy_type == "Docile":
            return DocilePlayer(name)
        elif self.strategy_type == "EarlyBidder":
            return EarlyBidder(name)
        else:
            return RandomPlayer(name)  # fallback
    
    def change_strategy(self, new_strategy):
        """Change this agent's strategy"""
        self.strategy_type = new_strategy
        self.strategy_history.append(new_strategy)
        print(f"  Agent{self.agent_id} mutates to {new_strategy}!")
    
    def record_game(self, won):
        """Record the outcome of a game"""
        self.games_played += 1
        if won:
            self.total_wins += 1


class EvolutionarySimulation:
    """Runs evolutionary simulation where agents can change strategies"""
    
    def __init__(self, 
                 population_size=20,
                 initial_strategy="Random",
                 available_strategies=None,
                 mutation_rate=0.3,
                 bottom_x=2):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.bottom_x = bottom_x  # how many bottom players can mutate
        
        if available_strategies is None:
            self.available_strategies = ["Random", "Aggressive", "Pacifist", "Docile", "EarlyBidder"]
        else:
            self.available_strategies = available_strategies
        
        # Initialize population
        self.agents = [
            EvolutionaryAgent(i, initial_strategy) 
            for i in range(population_size)
        ]
        
        # Tracking
        self.game_count = 0
        self.strategy_distribution_history = []
        self.winner_history = []
        
    def select_game_participants(self):
        """Randomly select 4 agents ensuring equal play distribution"""
        # Try to select agents with fewest games played first
        sorted_agents = sorted(self.agents, key=lambda a: a.games_played)
        
        # Select 4 agents, biased toward those with fewer games
        # Use weighted random to balance fairness and randomness
        weights = [1.0 / (a.games_played + 1) for a in self.agents]
        selected = random.choices(self.agents, weights=weights, k=4)
        
        return selected
    
    def play_game(self, agent_list):
        """Play one game with the given agents and return rankings"""
        # Create player instances for each agent
        players = [agent.create_player_instance() for agent in agent_list]
        
        # Map players back to agents for tracking
        player_to_agent = {player: agent for player, agent in zip(players, agent_list)}
        
        # Run the game
        game = SkullGame(players)
        
        # Run game manually to track rounds
        while (
            len([p for p in game.players if p.active and p.hand]) > 1
            and not any(p.score >= 2 for p in game.players)
        ):
            game.play_round()
        
        # Determine winner
        active_players = [p for p in game.players if p.active and p.hand]
        if len(active_players) == 1:
            winner = active_players[0]
        else:
            winner = max(game.players, key=lambda p: p.score)
        
        # Rank players: winner first, then by score, then by cards remaining
        def rank_key(player):
            return (
                -player.score,  # higher score is better
                -len(player.hand),  # more cards is better
                not player.active  # active is better
            )
        
        ranked_players = sorted(players, key=rank_key)
        ranked_agents = [player_to_agent[p] for p in ranked_players]
        
        return winner, ranked_agents, player_to_agent[winner]
    
    def apply_evolution(self, ranked_agents, winner_agent):
        """Apply mutation to bottom performers - mostly copy winner, sometimes explore"""
        for i in range(len(ranked_agents) - self.bottom_x, len(ranked_agents)):
            agent = ranked_agents[i]
            
            if random.random() < self.mutation_rate:
                # 60% copy winner, 40% random exploration
                if random.random() < 0.6:
                    new_strategy = winner_agent.strategy_type
                else:
                    new_strategy = random.choice(self.available_strategies)
                
                if new_strategy != agent.strategy_type:
                    agent.change_strategy(new_strategy)
    
    def record_game_results(self, ranked_agents, winner_agent):
        """Record statistics from the game"""
        # Record participation and wins
        for agent in ranked_agents:
            won = (agent == winner_agent)
            agent.record_game(won)
        
        # Track winner
        self.winner_history.append({
            'game': self.game_count,
            'agent_id': winner_agent.agent_id,
            'strategy': winner_agent.strategy_type
        })
        
        # Track strategy distribution
        distribution = self.get_strategy_distribution()
        self.strategy_distribution_history.append({
            'game': self.game_count,
            'distribution': distribution.copy()
        })
    
    def get_strategy_distribution(self):
        """Get current count of each strategy in population"""
        distribution = {strategy: 0 for strategy in self.available_strategies}
        for agent in self.agents:
            distribution[agent.strategy_type] += 1
        return distribution
    
    def check_termination(self, max_games=None, win_threshold=None, convergence_threshold=None, min_games=100):
        """Check if simulation should terminate"""
        # Need minimum games before checking convergence (to allow evolution to happen)
        if self.game_count < min_games:
            return False
        
        # Max games reached
        if max_games and self.game_count >= max_games:
            print(f"\nTermination: Reached maximum games ({max_games})")
            return True
        
        # Win threshold reached
        if win_threshold:
            max_wins = max(agent.total_wins for agent in self.agents)
            if max_wins >= win_threshold:
                winner = max(self.agents, key=lambda a: a.total_wins)
                print(f"\nTermination: Agent{winner.agent_id} reached {max_wins} wins (threshold: {win_threshold})")
                return True
        
        # Convergence threshold reached (only after min_games)
        if convergence_threshold:
            distribution = self.get_strategy_distribution()
            max_count = max(distribution.values())
            if max_count / self.population_size >= convergence_threshold:
                dominant = max(distribution, key=distribution.get)
                print(f"\nTermination: Population converged to {dominant} ({max_count}/{self.population_size})")
                return True
        
        return False
    
    def run_simulation(self, max_games=1000, win_threshold=50, convergence_threshold=0.9, verbose=True):
        """Run the evolutionary simulation"""
        print("=== STARTING EVOLUTIONARY SIMULATION ===")
        print(f"Population: {self.population_size} agents")
        print(f"Initial strategy: {self.agents[0].strategy_type}")
        print(f"Available strategies: {self.available_strategies}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Bottom {self.bottom_x} players can mutate\n")
        
        while not self.check_termination(max_games, win_threshold, convergence_threshold):
            self.game_count += 1
            
            # Select participants
            participants = self.select_game_participants()
            
            if verbose and self.game_count % 50 == 0:
                print(f"\n--- Game {self.game_count} ---")
                dist = self.get_strategy_distribution()
                print(f"Strategy distribution: {dist}")
            
            # Play game
            winner_player, ranked_agents, winner_agent = self.play_game(participants)
            
            # Record results
            self.record_game_results(ranked_agents, winner_agent)
            
            if verbose:
                strategies = [f"Agent{a.agent_id}({a.strategy_type})" for a in participants]
                print(f"Game {self.game_count}: {', '.join(strategies)} -> Winner: Agent{winner_agent.agent_id}")
            
            # Apply evolution
            self.apply_evolution(ranked_agents, winner_agent)
        
        self.print_final_report()
    
    def print_final_report(self):
        """Print final statistics"""
        print("\n" + "="*60)
        print("EVOLUTIONARY SIMULATION COMPLETE")
        print("="*60)
        
        print(f"\nTotal games played: {self.game_count}")
        
        # Final strategy distribution
        print("\n--- Final Strategy Distribution ---")
        distribution = self.get_strategy_distribution()
        for strategy, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.population_size) * 100
            print(f"{strategy:12}: {count:2} agents ({percentage:5.1f}%)")
        
        # Top performers
        print("\n--- Top 5 Agents by Wins ---")
        top_agents = sorted(self.agents, key=lambda a: a.total_wins, reverse=True)[:5]
        for agent in top_agents:
            win_rate = (agent.total_wins / agent.games_played * 100) if agent.games_played > 0 else 0
            print(f"Agent{agent.agent_id:2} ({agent.strategy_type:12}): "
                  f"{agent.total_wins:3} wins / {agent.games_played:3} games ({win_rate:5.1f}%)")
        
        # Strategy performance
        print("\n--- Strategy Performance ---")
        strategy_stats = {s: {'wins': 0, 'games': 0} for s in self.available_strategies}
        for agent in self.agents:
            strategy_stats[agent.strategy_type]['wins'] += agent.total_wins
            strategy_stats[agent.strategy_type]['games'] += agent.games_played
        
        for strategy in sorted(self.available_strategies):
            stats = strategy_stats[strategy]
            if stats['games'] > 0:
                win_rate = (stats['wins'] / stats['games']) * 100
                print(f"{strategy:12}: {stats['wins']:4} wins / {stats['games']:4} games ({win_rate:5.1f}%)")
            else:
                print(f"{strategy:12}: No games played")
        
        # Strategy changes
        print("\n--- Strategy Evolution ---")
        total_changes = sum(len(agent.strategy_history) - 1 for agent in self.agents)
        print(f"Total strategy changes: {total_changes}")
        
        most_changed = max(self.agents, key=lambda a: len(a.strategy_history))
        print(f"Most changed: Agent{most_changed.agent_id} ({len(most_changed.strategy_history)-1} changes)")
        print(f"  Path: {' -> '.join(most_changed.strategy_history)}")


if __name__ == "__main__":
    # Run simulation
    sim = EvolutionarySimulation(
        population_size=20,
        initial_strategy="Random",
        available_strategies=["Random", "Aggressive", "Pacifist", "Docile", "EarlyBidder"],
        mutation_rate=0.3,
        bottom_x=2
    )
    
    sim.run_simulation(
        max_games=10000,
        win_threshold=10000,
        convergence_threshold=0.95,
        verbose=False  
    )
import random
from SkullGame import SkullGame
from RandomPlayer import RandomPlayer
from DocilePlayer import DocilePlayer
from PacifistPlayer import PacifistPlayer
from AggressivePlayer import AggressivePlayer
from EarlyBidder import EarlyBidder
import matplotlib.pyplot as plt
import numpy as np

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
            return RandomPlayer(name)
    
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
                 bottom_x=2,
                 initial_temperature=1.0,
                 final_temperature=0.1,
                 temperature_decay='linear'):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.bottom_x = bottom_x
        
        # Temperature parameters for simulated annealing
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_decay = temperature_decay  # 'linear', 'exponential', or 'step'
        self.current_temperature = initial_temperature
        
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
        self.temperature_history = []  # Track temperature over time
        
        # Beat matrix tracking
        self.beat_matrix = {s: {t: {'wins': 0, 'games': 0} for t in self.available_strategies} 
                           for s in self.available_strategies}
        self.elimination_matrix = {s: {t: 0 for t in self.available_strategies} 
                                  for s in self.available_strategies}
        
    def select_game_participants(self):
        """Randomly select 4 agents ensuring equal play distribution"""
        weights = [1.0 / (a.games_played + 1) for a in self.agents]
        selected = random.choices(self.agents, weights=weights, k=4)
        return selected
    
    def play_game(self, agent_list):
        """Play one game with the given agents and return rankings"""
        players = [agent.create_player_instance() for agent in agent_list]
        player_to_agent = {player: agent for player, agent in zip(players, agent_list)}
        
        game = SkullGame(players)
        
        while (
            len([p for p in game.players if p.active and p.hand]) > 1
            and not any(p.score >= 2 for p in game.players)
        ):
            game.play_round()
        
        active_players = [p for p in game.players if p.active and p.hand]
        if len(active_players) == 1:
            winner = active_players[0]
        else:
            winner = max(game.players, key=lambda p: p.score)
        
        def rank_key(player):
            return (-player.score, -len(player.hand), not player.active)
        
        ranked_players = sorted(players, key=rank_key)
        ranked_agents = [player_to_agent[p] for p in ranked_players]
        
        return winner, ranked_agents, player_to_agent[winner]
    
    def apply_evolution(self, ranked_agents, winner_agent):
        """Apply mutation to bottom performers with temperature-based selection"""
        for i in range(len(ranked_agents) - self.bottom_x, len(ranked_agents)):
            agent = ranked_agents[i]
            
            if random.random() < self.mutation_rate:
                # Temperature controls exploration vs exploitation
                # Higher temp = more random, lower temp = more winner-copying
                
                # Calculate probability of copying winner based on temperature
                # At temp=1.0: 60% copy winner, 40% explore
                # At temp=0.1: 95% copy winner, 5% explore
                base_copy_prob = 0.6
                temp_adjusted_copy_prob = base_copy_prob + (1 - base_copy_prob) * (1 - self.current_temperature)
                
                if random.random() < temp_adjusted_copy_prob:
                    new_strategy = winner_agent.strategy_type
                else:
                    new_strategy = random.choice(self.available_strategies)
                
                if new_strategy != agent.strategy_type:
                    agent.change_strategy(new_strategy)
    
    def update_temperature(self, progress):
        """
        Update temperature based on simulation progress.
        progress: float between 0.0 (start) and 1.0 (end)
        """
        temp_range = self.initial_temperature - self.final_temperature
        
        if self.temperature_decay == 'linear':
            # Linear decay: T = T_initial - (T_initial - T_final) * progress
            self.current_temperature = self.initial_temperature - (temp_range * progress)
        
        elif self.temperature_decay == 'exponential':
            # Exponential decay: faster cooling early, slower later
            # T = T_final + (T_initial - T_final) * e^(-5*progress)
            import math
            decay_rate = 5
            self.current_temperature = self.final_temperature + temp_range * math.exp(-decay_rate * progress)
        
        elif self.temperature_decay == 'step':
            # Step function: high temp for first 25%, then drop
            if progress < 0.25:
                self.current_temperature = self.initial_temperature
            elif progress < 0.50:
                self.current_temperature = self.initial_temperature * 0.5
            elif progress < 0.75:
                self.current_temperature = self.final_temperature * 2
            else:
                self.current_temperature = self.final_temperature
        
        else:
            # Default to linear
            self.current_temperature = self.initial_temperature - (temp_range * progress)
    
    def record_game_results(self, ranked_agents, winner_agent):
        """Record statistics from the game"""
        for agent in ranked_agents:
            won = (agent == winner_agent)
            agent.record_game(won)
        
        self.winner_history.append({
            'game': self.game_count,
            'agent_id': winner_agent.agent_id,
            'strategy': winner_agent.strategy_type
        })
        
        distribution = self.get_strategy_distribution()
        self.strategy_distribution_history.append({
            'game': self.game_count,
            'distribution': distribution.copy()
        })
        
        # Track temperature
        self.temperature_history.append({
            'game': self.game_count,
            'temperature': self.current_temperature
        })
        
        # Update beat matrix
        self._update_beat_matrix(ranked_agents, winner_agent)
    
    def get_strategy_distribution(self):
        """Get current count of each strategy in population"""
        distribution = {strategy: 0 for strategy in self.available_strategies}
        for agent in self.agents:
            distribution[agent.strategy_type] += 1
        return distribution
    
    def check_termination(self, max_games=None, win_threshold=None, convergence_threshold=None, 
                     min_games=100, require_stability=False, stability_window=100):
        """Check if simulation should terminate"""
        if self.game_count < min_games:
            return False
        
        # Check for stability if required
        if require_stability and self.game_count >= min_games:
            is_stable, score = self.calculate_stability(window=stability_window)
            quality = self.analyze_convergence_quality(window=stability_window)
            
            if is_stable and quality and quality['is_stable']:
                print(f"\nTermination: Population is stable!")
                print(f"  Stability score: {score:.3f}")
                print(f"  Variance: {quality['total_variance']:.3f}")
                print(f"  Recent change: {quality['total_change_in_window']} agents")
                return True
        
        if max_games and self.game_count >= max_games:
            print(f"\nTermination: Reached maximum games ({max_games})")
            if require_stability:
                is_stable, score = self.calculate_stability(window=stability_window)
                print(f"  Warning: Stability not reached (score: {score:.3f})")
            return True
        
        if win_threshold:
            max_wins = max(agent.total_wins for agent in self.agents)
            if max_wins >= win_threshold:
                winner = max(self.agents, key=lambda a: a.total_wins)
                print(f"\nTermination: Agent{winner.agent_id} reached {max_wins} wins")
                return True
        
        if convergence_threshold and not require_stability:
            distribution = self.get_strategy_distribution()
            max_count = max(distribution.values())
            if max_count / self.population_size >= convergence_threshold:
                dominant = max(distribution, key=distribution.get)
                print(f"\nTermination: Population converged to {dominant} ({max_count}/{self.population_size})")
                return True
        
        return False
    
    def run_simulation(self, max_games=5000, win_threshold=None, convergence_threshold=None, 
                      min_games=200, require_stability=True, stability_window=100, verbose=True):
        """Run the evolutionary simulation"""
        print("=== STARTING EVOLUTIONARY SIMULATION ===")
        print(f"Population: {self.population_size} agents")
        print(f"Initial strategy: {self.agents[0].strategy_type}")
        print(f"Available strategies: {self.available_strategies}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Bottom {self.bottom_x} players can mutate")
        print(f"Temperature: {self.initial_temperature} → {self.final_temperature} ({self.temperature_decay} decay)")
        if require_stability:
            print(f"Will run until stable (min {min_games} games, max {max_games} games)")
            print(f"Stability window: {stability_window} games\n")
        else:
            print(f"Max games: {max_games}\n")
        
        while not self.check_termination(
            max_games=max_games, 
            win_threshold=win_threshold, 
            convergence_threshold=convergence_threshold, 
            min_games=min_games, 
            require_stability=require_stability, 
            stability_window=stability_window
        ):
            self.game_count += 1
            
            # Update temperature based on progress (use max_games for scaling)
            progress = min(self.game_count / max_games, 1.0)
            self.update_temperature(progress)
            
            participants = self.select_game_participants()
            
            if verbose and self.game_count % 50 == 0:
                print(f"\n--- Game {self.game_count} ---")
                dist = self.get_strategy_distribution()
                print(f"Strategy distribution: {dist}")
                print(f"Temperature: {self.current_temperature:.3f}")
                
                # Show stability status
                if self.game_count >= stability_window:
                    is_stable, score = self.calculate_stability(window=stability_window)
                    print(f"Stability: {'✓ STABLE' if is_stable else 'EVOLVING'} (score: {score:.3f})")
            
            winner_player, ranked_agents, winner_agent = self.play_game(participants)
            self.record_game_results(ranked_agents, winner_agent)
            
            if verbose and self.game_count % 100 == 0:
                strategies = [f"Agent{a.agent_id}({a.strategy_type})" for a in participants]
                print(f"Game {self.game_count}: {', '.join(strategies)} -> Winner: Agent{winner_agent.agent_id}")
            
            self.apply_evolution(ranked_agents, winner_agent)
        
        self.print_final_report()
        self.generate_full_report(save_plots=True)
        
        self.print_final_report()
        self.generate_full_report(save_plots=True)
    
    # ========================================================================
    # BEAT MATRIX TRACKING
    # ========================================================================
    
    def _update_beat_matrix(self, ranked_agents, winner_agent):
        """Update beat matrix after each game"""
        winner_strategy = winner_agent.strategy_type
        
        # Track all pairwise matchups in this game
        # We need to record from BOTH perspectives
        for agent in ranked_agents:
            agent_strategy = agent.strategy_type
            
            # For every other agent in the game
            for other_agent in ranked_agents:
                if agent != other_agent:
                    other_strategy = other_agent.strategy_type
                    
                    # Record that these two strategies played against each other
                    self.beat_matrix[agent_strategy][other_strategy]['games'] += 1
                    
                    # Record if this agent won (beat the other agent)
                    if agent == winner_agent:
                        self.beat_matrix[agent_strategy][other_strategy]['wins'] += 1
            
            # Track eliminations (last place)
            if agent == ranked_agents[-1] and agent != winner_agent:
                self.elimination_matrix[winner_strategy][agent.strategy_type] += 1
    
    def print_beat_matrix(self):
        """Print beat matrix showing head-to-head win rates"""
        print("\n" + "="*70)
        print("BEAT MATRIX - Win Rate When X Plays Against Y")
        print("="*70)
        print("\nRows = Winner Strategy | Columns = Opponent Strategy")
        print("Shows: Wins / Games (Win %)\n")
        
        # Print header
        col_width = 15
        print(f"{'':>{col_width}}", end="")
        for strategy in self.available_strategies:
            print(f"{strategy[:12]:>{col_width}}", end="")
        print()
        print("-" * (col_width * (len(self.available_strategies) + 1)))
        
        # Print each row
        for strat_row in self.available_strategies:
            print(f"{strat_row[:12]:>{col_width}}", end="")
            for strat_col in self.available_strategies:
                data = self.beat_matrix[strat_row][strat_col]
                wins = data['wins']
                games = data['games']
                
                if games > 0:
                    win_rate = (wins / games) * 100
                    if strat_row == strat_col:
                        # Same strategy - show games played
                        print(f"{'--':>{col_width}}", end="")
                    else:
                        print(f"{wins}/{games} ({win_rate:.0f}%):>{col_width}"[:col_width], end="")
                else:
                    print(f"{'0/0':>{col_width}}", end="")
            print()
        
        print("\n" + "="*70)
    
    def print_elimination_matrix(self):
        """Print matrix showing which strategies eliminate which"""
        print("\n" + "="*70)
        print("ELIMINATION MATRIX - Who Eliminates Who")
        print("="*70)
        print("\nRows = Strategy | Columns = Strategies They Eliminated\n")
        
        col_width = 12
        print(f"{'':>{col_width}}", end="")
        for strategy in self.available_strategies:
            print(f"{strategy[:10]:>{col_width}}", end="")
        print()
        print("-" * (col_width * (len(self.available_strategies) + 1)))
        
        for strat_row in self.available_strategies:
            print(f"{strat_row[:10]:>{col_width}}", end="")
            for strat_col in self.available_strategies:
                count = self.elimination_matrix[strat_row][strat_col]
                if count > 0:
                    print(f"{count:>{col_width}}", end="")
                else:
                    print(f"{'-':>{col_width}}", end="")
            print()
        
        print("\n" + "="*70)
    
    def analyze_strategy_matchups(self):
        """Analyze and print key matchup insights"""
        print("\n" + "="*70)
        print("STRATEGY MATCHUP ANALYSIS")
        print("="*70)
        
        # Find best and worst matchups for each strategy
        for strategy in self.available_strategies:
            print(f"\n{strategy}:")
            
            matchups = []
            for opponent in self.available_strategies:
                if strategy != opponent:
                    data = self.beat_matrix[strategy][opponent]
                    if data['games'] >= 5:  # Only consider matchups with 5+ games
                        win_rate = (data['wins'] / data['games']) * 100
                        matchups.append((opponent, win_rate, data['wins'], data['games']))
            
            if matchups:
                matchups.sort(key=lambda x: x[1], reverse=True)
                
                # Best matchup
                best = matchups[0]
                print(f"  Best vs: {best[0]} ({best[1]:.0f}% - {best[2]}/{best[3]} games)")
                
                # Worst matchup
                worst = matchups[-1]
                print(f"  Worst vs: {worst[0]} ({worst[1]:.0f}% - {worst[2]}/{worst[3]} games)")
                
                # Overall win rate against others
                total_wins = sum(m[2] for m in matchups)
                total_games = sum(m[3] for m in matchups)
                if total_games > 0:
                    overall = (total_wins / total_games) * 100
                    print(f"  Overall vs others: {overall:.1f}% ({total_wins}/{total_games})")
            else:
                print(f"  Not enough data (need 5+ games per matchup)")
    
    def plot_beat_matrix_heatmap(self, save_path=None):
        """Create heatmap visualization of beat matrix"""
        
        # Build win rate matrix
        n = len(self.available_strategies)
        win_rates = np.zeros((n, n))
        
        for i, strat_row in enumerate(self.available_strategies):
            for j, strat_col in enumerate(self.available_strategies):
                data = self.beat_matrix[strat_row][strat_col]
                if data['games'] > 0 and strat_row != strat_col:
                    win_rates[i, j] = (data['wins'] / data['games']) * 100
                else:
                    win_rates[i, j] = np.nan  # Don't show same-strategy matchups
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(win_rates, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.available_strategies, rotation=45, ha='right')
        ax.set_yticklabels(self.available_strategies)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Win Rate (%)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                if not np.isnan(win_rates[i, j]):
                    data = self.beat_matrix[self.available_strategies[i]][self.available_strategies[j]]
                    text = f"{win_rates[i, j]:.0f}%\n({data['wins']}/{data['games']})"
                    ax.text(j, i, text, ha='center', va='center', 
                           color='black' if 30 < win_rates[i, j] < 70 else 'white',
                           fontsize=9)
        
        ax.set_xlabel('Opponent Strategy', fontsize=12)
        ax.set_ylabel('Winner Strategy', fontsize=12)
        ax.set_title('Beat Matrix: Win Rate in Head-to-Head Matchups', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Beat matrix heatmap saved to {save_path}")
        
        plt.show()
    
    # ========================================================================
    # EGT TRACKING AND ANALYSIS METHODS
    # ========================================================================
    
    def calculate_stability(self, window=100):
        """Calculate stability metric - lower = more stable"""
        if len(self.strategy_distribution_history) < window:
            return False, float('inf')
        
        recent = self.strategy_distribution_history[-window:]
        total_change = 0
        
        for i in range(len(recent) - 1):
            for strategy in self.available_strategies:
                change = abs(recent[i+1]['distribution'][strategy] - 
                            recent[i]['distribution'][strategy])
                total_change += change
        
        avg_change = total_change / (window - 1)
        is_stable = avg_change < 0.5
        
        return is_stable, avg_change
    
    def detect_convergence_pattern(self):
        """Detect what kind of equilibrium emerged"""
        if not self.strategy_distribution_history:
            return "No data"
        
        final_dist = self.strategy_distribution_history[-1]['distribution']
        total = sum(final_dist.values())
        proportions = {s: (final_dist[s] / total * 100) for s in self.available_strategies}
        dominant = max(proportions.items(), key=lambda x: x[1])
        
        if dominant[1] >= 90:
            return f"DOMINANT: {dominant[0]} ({dominant[1]:.1f}%)"
        elif dominant[1] >= 60:
            return f"MAJORITY: {dominant[0]} ({dominant[1]:.1f}%)"
        else:
            active = [s for s, p in proportions.items() if p >= 10]
            return f"MIXED: {len(active)} strategies coexist"
    
    def detect_convergence_point(self, stability_threshold=0.3, window=50):
        """
        Detect at which game the population converged to its final state.
        Returns (converged: bool, convergence_game: int, final_distribution: dict)
        """
        if len(self.strategy_distribution_history) < window * 2:
            return False, None, None
        
        # Work backwards from the end to find when population stabilized
        final_dist = self.strategy_distribution_history[-1]['distribution']
        
        for i in range(len(self.strategy_distribution_history) - window, window, -1):
            current_dist = self.strategy_distribution_history[i]['distribution']
            
            # Check if distribution is similar to final state
            total_diff = sum(abs(final_dist[s] - current_dist[s]) for s in self.available_strategies)
            
            if total_diff > stability_threshold * self.population_size:
                # This is where it started to stabilize
                convergence_game = self.strategy_distribution_history[i + 1]['game']
                return True, convergence_game, final_dist
        
        # Converged from the start (unlikely)
        return True, 0, final_dist
    
    def analyze_convergence_quality(self, window=50):
        """
        Analyze the quality and stability of convergence.
        Returns dict with convergence metrics.
        """
        if len(self.strategy_distribution_history) < window:
            return None
        
        recent = self.strategy_distribution_history[-window:]
        
        # Calculate variance for each strategy over the window
        variances = {}
        for strategy in self.available_strategies:
            counts = [entry['distribution'][strategy] for entry in recent]
            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            variances[strategy] = variance
        
        total_variance = sum(variances.values())
        
        # Calculate how much the distribution changed in the last window
        first_in_window = recent[0]['distribution']
        last_in_window = recent[-1]['distribution']
        total_change = sum(abs(last_in_window[s] - first_in_window[s]) 
                          for s in self.available_strategies)
        
        return {
            'total_variance': total_variance,
            'strategy_variances': variances,
            'total_change_in_window': total_change,
            'is_stable': total_variance < 2.0 and total_change < 3
        }
    
    def get_convergence_distribution(self):
        """
        Get the final converged distribution with detailed stats.
        Returns dict with percentages and confidence metrics.
        """
        if not self.strategy_distribution_history:
            return None
        
        final_dist = self.strategy_distribution_history[-1]['distribution']
        total = sum(final_dist.values())
        
        result = {}
        for strategy in self.available_strategies:
            count = final_dist[strategy]
            percentage = (count / total * 100) if total > 0 else 0
            result[strategy] = {
                'count': count,
                'percentage': percentage,
                'is_present': count > 0,
                'is_dominant': percentage >= 50,
                'is_significant': percentage >= 10
            }
        
        return result
    
    def plot_strategy_evolution(self, save_path=None):
        """Plot strategy distribution over time"""
        games = [e['game'] for e in self.strategy_distribution_history]
        strategy_data = {s: [] for s in self.available_strategies}
        
        for entry in self.strategy_distribution_history:
            for strategy in self.available_strategies:
                strategy_data[strategy].append(entry['distribution'][strategy])
        
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax4 = fig.add_subplot(gs[2, :])
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        # Stacked area
        ax1.stackplot(games, *[strategy_data[s] for s in self.available_strategies],
                      labels=self.available_strategies,
                      colors=colors[:len(self.available_strategies)], alpha=0.8)
        ax1.set_xlabel('Game Number', fontsize=12)
        ax1.set_ylabel('Number of Agents', fontsize=12)
        ax1.set_title('Population Over Time (Stacked)', fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Line chart
        for i, strategy in enumerate(self.available_strategies):
            ax2.plot(games, strategy_data[strategy], label=strategy, 
                    linewidth=2, color=colors[i % len(colors)])
        ax2.set_xlabel('Game Number', fontsize=12)
        ax2.set_ylabel('Number of Agents', fontsize=12)
        ax2.set_title('Strategy Trajectories', fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Proportions over time
        for i, strategy in enumerate(self.available_strategies):
            proportions = [(strategy_data[strategy][j] / self.population_size * 100) 
                          for j in range(len(games))]
            ax3.plot(games, proportions, label=strategy, 
                    linewidth=2, color=colors[i % len(colors)])
        ax3.set_xlabel('Game Number', fontsize=12)
        ax3.set_ylabel('Percentage (%)', fontsize=12)
        ax3.set_title('Strategy Proportions Over Time', fontweight='bold')
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Temperature over time
        temp_games = [e['game'] for e in self.temperature_history]
        temps = [e['temperature'] for e in self.temperature_history]
        ax4.plot(temp_games, temps, linewidth=2, color='darkred')
        ax4.set_xlabel('Game Number', fontsize=12)
        ax4.set_ylabel('Temperature', fontsize=12)
        ax4.set_title('Imitation Temperature (Exploration vs Exploitation)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(self.initial_temperature, 1.1))
        
        # Add annotations
        if temps:
            ax4.axhline(y=self.initial_temperature, color='orange', linestyle='--', 
                       alpha=0.5, label=f'Initial: {self.initial_temperature}')
            ax4.axhline(y=self.final_temperature, color='blue', linestyle='--', 
                       alpha=0.5, label=f'Final: {self.final_temperature}')
            ax4.legend(loc='upper right')
        
        plt.suptitle('Evolutionary Dynamics with Temperature Annealing', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_full_report(self, save_plots=True):
        """Generate comprehensive analysis"""
        print("\n" + "="*60)
        print("EGT ANALYSIS")
        print("="*60)
        
        # Stability
        is_stable, score = self.calculate_stability(min(100, self.game_count // 2))
        print(f"\nStability Score: {score:.3f}")
        print(f"Status: {'STABLE ✓' if is_stable else 'EVOLVING...'}")
        
        # Pattern
        print(f"\nConvergence: {self.detect_convergence_pattern()}")
        
        # Convergence point detection
        self.print_convergence_analysis()
        
        # Beat matrix analysis
        self.print_beat_matrix()
        self.analyze_strategy_matchups()
        self.print_elimination_matrix()
        
        # Plots
        if save_plots:
            self.plot_strategy_evolution('strategy_evolution.png')
            self.plot_beat_matrix_heatmap('beat_matrix_heatmap.png')
    
    def print_convergence_analysis(self):
        """Print detailed convergence analysis"""
        print("\n" + "="*70)
        print("CONVERGENCE ANALYSIS")
        print("="*70)
        
        # Detect convergence point
        converged, conv_game, final_dist = self.detect_convergence_point()
        
        if converged and conv_game is not None:
            print(f"\nConvergence detected at game: {conv_game}")
            print(f"Time to convergence: {conv_game} games")
            print(f"Stability period: {self.game_count - conv_game} games")
        else:
            print("\nNo clear convergence point detected")
            print("Population may still be evolving")
        
        # Convergence quality
        quality = self.analyze_convergence_quality()
        if quality:
            print(f"\nConvergence Quality:")
            print(f"  Total variance: {quality['total_variance']:.3f}")
            print(f"  Recent change: {quality['total_change_in_window']} agents")
            print(f"  Stability: {'HIGH ✓' if quality['is_stable'] else 'LOW - still fluctuating'}")
            
            print(f"\n  Strategy-specific variance:")
            for strategy, var in sorted(quality['strategy_variances'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"    {strategy:12}: {var:.3f}")
        
        # Final equilibrium distribution
        conv_dist = self.get_convergence_distribution()
        if conv_dist:
            print(f"\n" + "="*70)
            print("FINAL EQUILIBRIUM DISTRIBUTION")
            print("="*70)
            
            # Sort by percentage
            sorted_strategies = sorted(conv_dist.items(), 
                                      key=lambda x: x[1]['percentage'], 
                                      reverse=True)
            
            for strategy, data in sorted_strategies:
                bar = '█' * int(data['percentage'] / 5)
                status = []
                if data['is_dominant']:
                    status.append('DOMINANT')
                elif data['is_significant']:
                    status.append('SIGNIFICANT')
                if data['is_present']:
                    status_str = f" [{', '.join(status)}]" if status else ""
                else:
                    status_str = " [EXTINCT]"
                
                print(f"  {strategy:12}: {data['count']:2} agents ({data['percentage']:5.1f}%) {bar}{status_str}")
            
            # Categorize equilibrium type
            dominant_count = sum(1 for d in conv_dist.values() if d['is_dominant'])
            significant_count = sum(1 for d in conv_dist.values() if d['is_significant'])
            
            print(f"\nEquilibrium Type:")
            if dominant_count == 1:
                print(f"  → SINGLE DOMINANT STRATEGY (ESS candidate)")
            elif significant_count >= 3:
                print(f"  → DIVERSE MIXED EQUILIBRIUM ({significant_count} strategies)")
            elif significant_count == 2:
                print(f"  → STABLE DUOPOLY (2 strategies coexist)")
            else:
                print(f"  → UNCLEAR EQUILIBRIUM (may need more time)")
        
        print("\n" + "="*70)
    
    def print_final_report(self):
        """Print final statistics"""
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        
        print(f"\nTotal games: {self.game_count}")
        
        print("\n--- Final Distribution ---")
        distribution = self.get_strategy_distribution()
        for strategy, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.population_size) * 100
            bar = '█' * int(pct / 5)
            print(f"{strategy:12}: {count:2} ({pct:5.1f}%) {bar}")
        
        print("\n--- Top 5 Agents ---")
        top = sorted(self.agents, key=lambda a: a.total_wins, reverse=True)[:5]
        for agent in top:
            wr = (agent.total_wins / agent.games_played * 100) if agent.games_played > 0 else 0
            print(f"Agent{agent.agent_id:2} ({agent.strategy_type:12}): "
                  f"{agent.total_wins:3}W / {agent.games_played:3}G ({wr:5.1f}%)")
        
        print("\n--- Strategy Performance ---")
        stats = {s: {'wins': 0, 'games': 0} for s in self.available_strategies}
        for agent in self.agents:
            stats[agent.strategy_type]['wins'] += agent.total_wins
            stats[agent.strategy_type]['games'] += agent.games_played
        
        for strategy in sorted(self.available_strategies):
            s = stats[strategy]
            if s['games'] > 0:
                wr = (s['wins'] / s['games']) * 100
                print(f"{strategy:12}: {s['wins']:4}W / {s['games']:4}G ({wr:5.1f}%)")


if __name__ == "__main__":
    sim = EvolutionarySimulation(
        population_size=20,
        initial_strategy="Random",
        available_strategies=["Random", "Aggressive", "Pacifist", "Docile", "EarlyBidder"],
        mutation_rate=0.3,
        bottom_x=2,
        initial_temperature=1.0,  # High exploration at start
        final_temperature=0.1,    # Low exploration at end
        temperature_decay='exponential'  # 'linear', 'exponential', or 'step'
    )
    
    sim.run_simulation(
        max_games=5000,              # Safety limit (won't run forever)
        min_games=200,               # Minimum games before checking stability
        require_stability=True,      # Run until stable (not just until max_games)
        stability_window=100,        # Check stability over last 100 games
        verbose=False                # Set True to see every 50 games
    )
import numpy as np
import matplotlib.pyplot as plt
import cv2

class RoboCupState:
    def __init__(self):
        # Field dimensions (in meters)
        self.field_length = 22.0  # Adjust based on your field size
        self.field_width = 14.0    # Adjust based on your field size
        
        # Initialize with example positions (you'll update these with real positions)
        self.our_positions = np.array([
            [-2.0, 1.0],   # Player 1 (ball holder)
            [-6.0, -1.5],  # Player 2
            [4.0, 0.0],    # Player 3
            [1.0, 5.0],    # Player 4
            [2.0, -2.0]    # Player 5
        ])
        
        self.opp_positions = np.array([
            [5.0, 5.0],    # Opponent 1
            [1.0, -1.0],   # Opponent 2
            [0.0, 1.0],    # Opponent 3
            [-1.0, -2.0],  # Opponent 4
            [-2.0, 2.0]    # Opponent 5
        ])
        
        self.our_velocities = np.array([
            [0.5, 0.0],    # Player 1 velocity
            [0.0, 0.5],    # Player 2 velocity
            [-0.5, 0.0],   # Player 3 velocity
            [0.0, -0.5],   # Player 4 velocity
            [0.3, 0.3]     # Player 5 velocity
        ])
        
        self.opp_velocities = np.array([
            [-0.3, 0.0],   # Opponent 1 velocity
            [0.0, -0.3],   # Opponent 2 velocity
            [0.3, 0.0],    # Opponent 3 velocity
            [0.0, 0.3],    # Opponent 4 velocity
            [-0.2, -0.2]   # Opponent 5 velocity
        ])
        
        self.ball_holder = 0  # Index of our robot holding the ball
        
        # Resolution for heat maps (pixels per meter)
        self.resolution = 20
        
        # Initialize grid
        self.x = np.linspace(-self.field_length/2, self.field_length/2, 
                           int(self.field_length * self.resolution))
        self.y = np.linspace(-self.field_width/2, self.field_width/2, 
                           int(self.field_width * self.resolution))
        self.X, self.Y = np.meshgrid(self.x, self.y)

class HeatMapGenerator:
    def __init__(self, state: RoboCupState):
        self.state = state
        
    def get_distance_from_point(self, point):
        """Calculate distance from each grid point to a specific point"""
        return np.sqrt((self.state.X - point[0])**2 + (self.state.Y - point[1])**2)
    
    def robots_repulsion_map(self, sigma=1.0):
        """Generate heat map where values increase away from all robots"""
        heat_map = np.zeros_like(self.state.X)
        
        # Add repulsion from our robots
        for pos in self.state.our_positions:
            distance = self.get_distance_from_point(pos)
            heat_map += 1 - np.exp(-distance**2 / (2*sigma**2))
            
        # Add repulsion from opponent robots
        for pos in self.state.opp_positions:
            distance = self.get_distance_from_point(pos)
            heat_map += 1 - np.exp(-distance**2 / (2*sigma**2))
            
        return heat_map / heat_map.max()  # Normalize
    
    def vertical_center_attraction_map(self):
        """Generate heat map with higher values near vertical center"""
        return 1 - np.abs(self.state.Y) / (self.state.field_width/2)
    
    def horizontal_right_attraction_map(self):
        # Create gradient using X coordinates
        # Normalize X coordinates to [0,1] range where:
        # leftmost = 0.0, rightmost = 1.0
        x_normalized = (self.state.X - self.state.X.min()) / (self.state.X.max() - self.state.X.min())
        return x_normalized
    
    def ball_holder_circle_map(self, radius=1.5):
        """Generate circular region around ball holder"""
        heat_map = np.zeros_like(self.state.X)
        
        # Create circles around all our robots with the ball holder having higher value
        for i, pos in enumerate(self.state.our_positions):
            distance = self.get_distance_from_point(pos)
            if i == self.state.ball_holder:
                heat_map += (distance <= radius).astype(float) * 1.0  # Full intensity for ball holder
                 
        return heat_map / heat_map.max()
    
    def ideal_pass_distance_map(self, A=1.0, r0=3.0, sigma=1.0):
        """Generate heat map based on ideal pass distance equation"""
        heat_map = np.zeros_like(self.state.X)
        
        # Calculate pass distance map from ball holder
        holder_pos = self.state.our_positions[self.state.ball_holder]
        r = self.get_distance_from_point(holder_pos)
        heat_map = A * np.exp(-(r - r0)**2 / (2*sigma**2))
            
        return heat_map / heat_map.max()
    
    def goal_direction_map(self, goal_pos=(10.2, 0.0), IGD=6.0, sigma=1.0, p=1.0):
        """
        Generate heat map based on goal probability equation:
        GoalProb = cos(α) * (p/(d*sqrt(2π))) * exp(-(dist_KG - IGD)^2 / (2σ^2))
        
        Parameters:
        - goal_pos: Position of the goal center (x, y)
        - IGD: Ideal Goal Distance
        - sigma: Standard deviation for the Gaussian distribution
        - p: Scaling parameter
        """
        holder_pos = self.state.our_positions[self.state.ball_holder]
        
        # Calculate angle component (cos(α))
        dx = self.state.X - holder_pos[0]
        dy = self.state.Y - holder_pos[1]
        
        # Calculate angles relative to ball holder
        angles = np.arctan2(dy, dx)
        goal_angle = np.arctan2(goal_pos[1] - holder_pos[1], 
                            goal_pos[0] - holder_pos[0])
        
        # Calculate cos(α) - angle difference from goal direction
        cos_alpha = np.cos(angles - goal_angle)
        
        # Calculate distance from each point to goal (dist_KG)
        dist_to_goal = np.sqrt((self.state.X - goal_pos[0])**2 + 
                            (self.state.Y - goal_pos[1])**2)
        
        # Calculate Gaussian component
        gaussian = np.exp(-(dist_to_goal - IGD)**2 / (2 * sigma**2))
        
        # Calculate normalization factor
        d = np.sqrt(dx**2 + dy**2)  # distance from ball holder to each point
        norm_factor = p / (d * np.sqrt(2 * np.pi))
        
        # Combine all components
        heat_map = cos_alpha * norm_factor * gaussian
        
        # Normalize to [0, 1] range
        heat_map = np.clip(heat_map, 0, None)  # Ensure non-negative values
        if heat_map.max() > 0:
            heat_map = heat_map / heat_map.max()
        
        return heat_map
    
    def combine_heat_maps(self, maps, weights=None):
        """Combine multiple heat maps with optional weights"""
        if weights is None:
            weights = [1.0] * len(maps)
        
        combined = np.zeros_like(self.state.X)
        for map_data, weight in zip(maps, weights):
            combined += weight * map_data
            
        return combined / combined.max()  # Normalize

class HeatMapVisualizer:
    def __init__(self, state: RoboCupState):
        self.state = state
        
    def show_matplotlib(self, heat_map, title="Heat Map"):
        """Display heat map using matplotlib with blue-red colormap"""
        plt.figure(figsize=(10, 8))
        
        # Create custom colormap from blue to red
        colors = ['darkblue', 'blue', 'royalblue', 'white', 'red', 'darkred']
        n_bins = 100
        cmap = plt.cm.RdBu_r  # Built-in blue-red colormap
        
        # Plot heatmap
        plt.imshow(heat_map, extent=[-self.state.field_length/2, self.state.field_length/2,
                                   -self.state.field_width/2, self.state.field_width/2],
                  origin='lower', cmap=cmap)
        
        # Plot robot positions
        plt.plot(self.state.our_positions[:, 0], self.state.our_positions[:, 1], 
                'go', label='Our Team', markersize=10)
        plt.plot(self.state.opp_positions[:, 0], self.state.opp_positions[:, 1], 
                'yo', label='Opponents', markersize=10)
        
        # Highlight ball holder
        plt.plot(self.state.our_positions[self.state.ball_holder, 0],
                self.state.our_positions[self.state.ball_holder, 1],
                'ro', markersize=15, label='Ball Holder')
        
        plt.colorbar(label='Value')
        plt.title(title)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_opencv_image(self, heat_map):
        """Convert heat map to OpenCV image format with blue-red colormap"""
        # Normalize to 0-255 range
        heat_map_normalized = (heat_map * 255).astype(np.uint8)
        
        # Apply custom colormap
        heat_map_color = cv2.applyColorMap(heat_map_normalized, cv2.COLORMAP_JET)
        
        # Convert to BGR for proper color display
        heat_map_color = cv2.cvtColor(heat_map_color, cv2.COLOR_RGB2BGR)
        
        return heat_map_color

# Example usage:
if __name__ == "__main__":
    # Initialize state
    state = RoboCupState()
    
    # Create generators and visualizer
    generator = HeatMapGenerator(state)
    visualizer = HeatMapVisualizer(state)
    
    # Generate individual heat maps
    repulsion_map = generator.robots_repulsion_map()
    vertical_map = generator.vertical_center_attraction_map()
    horizontal_map = generator.horizontal_right_attraction_map()  # New map
    ball_circle_map = generator.ball_holder_circle_map()
    pass_distance_map = generator.ideal_pass_distance_map()
    goal_map = generator.goal_direction_map()
    
    # Combine maps 
    combined_map = generator.combine_heat_maps(
        [repulsion_map, vertical_map, horizontal_map, ball_circle_map, pass_distance_map, goal_map],
        weights=[0.2, 0.15, 0.15, 0.2, 0.15, 0.15]  # Adjusted weights to include horizontal map
    )
    
 
  
    
    # Visualize individual maps
    visualizer.show_matplotlib(repulsion_map, "Robots Repulsion Map")
    visualizer.show_matplotlib(vertical_map, "Vertical Center Attraction Map")
    visualizer.show_matplotlib(horizontal_map, "Horizontal Right Attraction Map")
    visualizer.show_matplotlib(ball_circle_map, "Ball Holder Circle Map")
    visualizer.show_matplotlib(pass_distance_map, "Ideal Pass Distance Map")
    visualizer.show_matplotlib(goal_map, "Goal Direction Map")
    visualizer.show_matplotlib(combined_map, "Combined Heat Map")
    
    # Example of getting OpenCV image
    # cv_image = visualizer.get_opencv_image(combined_map)
    # cv2.imshow('Heat Map (OpenCV)', cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
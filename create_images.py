"""
Create placeholder images for rocket and target if you don't have them.
Run this script first before running the environment.
"""

import pygame
import sys

def create_rocket_image(filename="rocket.png", width=30, height=60):
    """Create a simple rocket sprite"""
    pygame.init()
    
    # Create surface with transparency
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((0, 0, 0, 0))  # Transparent background
    
    # Rocket body (green capsule)
    body_color = (50, 220, 80)
    nose_color = (40, 180, 60)
    fin_color = (30, 150, 50)
    engine_color = (200, 50, 50)
    
    # Main body
    pygame.draw.rect(surface, body_color, (8, 15, 14, 35))
    
    # Nose cone (top triangle)
    pygame.draw.polygon(surface, nose_color, [
        (width//2, 0),
        (5, 15),
        (width-5, 15)
    ])
    
    # Side fins
    pygame.draw.polygon(surface, fin_color, [
        (0, 35),
        (8, 25),
        (8, 45)
    ])
    pygame.draw.polygon(surface, fin_color, [
        (width, 35),
        (width-8, 25),
        (width-8, 45)
    ])
    
    # Engine nozzle
    pygame.draw.circle(surface, engine_color, (width//2, height-5), 6)
    pygame.draw.circle(surface, (100, 30, 30), (width//2, height-5), 3)
    
    # Windows
    pygame.draw.circle(surface, (150, 200, 255), (width//2, 20), 3)
    
    # Save
    pygame.image.save(surface, filename)
    print(f"✓ Created {filename}")
    
    pygame.quit()


def create_target_image(filename="target.png", width=240, height=80):
    """Create a landing pad target sprite"""
    pygame.init()
    
    # Create surface with transparency
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((0, 0, 0, 0))
    
    # Platform colors
    platform_color = (80, 120, 200)
    dark_color = (50, 80, 150)
    leg_color = (100, 100, 100)
    
    # Main platform
    pygame.draw.rect(surface, platform_color, (0, 0, width, height-20))
    
    # Platform details (darker stripe)
    pygame.draw.rect(surface, dark_color, (0, 10, width, 20))
    
    # Landing legs/wheels (circles at bottom)
    wheel_positions = [30, 70, 110, 150, 190, 210]
    for x in wheel_positions:
        pygame.draw.circle(surface, leg_color, (x, height-10), 12)
        pygame.draw.circle(surface, (60, 60, 60), (x, height-10), 8)
    
    # Center marking (target cross)
    center_x = width // 2
    center_y = height // 2 - 10
    cross_color = (255, 200, 50)
    
    # Horizontal line
    pygame.draw.line(surface, cross_color, 
                    (center_x - 30, center_y), 
                    (center_x + 30, center_y), 3)
    # Vertical line
    pygame.draw.line(surface, cross_color, 
                    (center_x, center_y - 15), 
                    (center_x, center_y + 15), 3)
    
    # Corner markers
    for corner_x in [20, width-20]:
        pygame.draw.circle(surface, (255, 255, 100), (corner_x, 15), 5)
    
    # Save
    pygame.image.save(surface, filename)
    print(f"✓ Created {filename}")
    
    pygame.quit()


def main():
    print("Creating placeholder images...")
    print("=" * 50)
    
    try:
        create_rocket_image("rocket.png", 30, 60)
        create_target_image("target.png", 240, 80)
        
        print("=" * 50)
        print("✓ All images created successfully!")
        print("\nYou can now run:")
        print("  - python run_rocket_env.py (to test environment)")
        print("  - python train_and_compare.py (to train agents)")
        
    except Exception as e:
        print(f"\n✗ Error creating images: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script to verify the image scaling functionality
"""

def scale_to_max_pixels(width: int, height: int, max_pixels: int = 768 * 768) -> tuple[int, int]:
    """Scale dimensions down proportionally if they exceed max_pixels while maintaining aspect ratio."""
    current_pixels = width * height
    
    if current_pixels <= max_pixels:
        return width, height
    
    # Calculate scaling factor to fit within max_pixels
    scale_factor = (max_pixels / current_pixels) ** 0.5
    
    # Apply scaling and round to integers
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    return scaled_width, scaled_height

def test_scaling():
    """Test various input dimensions to verify scaling behavior"""
    test_cases = [
        (1024, 1024),  # Should be scaled down (1,048,576 > 589,824)
        (768, 768),    # Should remain the same (589,824 = 589,824)
        (512, 512),    # Should remain the same (262,144 < 589,824)
        (1920, 1080),  # Should be scaled down (2,073,600 > 589,824)
        (800, 600),    # Should remain the same (480,000 < 589,824)
        (2048, 2048),  # Should be scaled down significantly (4,194,304 > 589,824)
        (1200, 800),   # Should be scaled down (960,000 > 589,824)
    ]
    
    max_pixels = 768 * 768
    print(f"Maximum allowed pixels: {max_pixels:,}")
    print("=" * 80)
    
    for original_w, original_h in test_cases:
        scaled_w, scaled_h = scale_to_max_pixels(original_w, original_h)
        original_pixels = original_w * original_h
        scaled_pixels = scaled_w * scaled_h
        
        print(f"Original: {original_w}x{original_h} ({original_pixels:,} pixels)")
        print(f"Scaled:   {scaled_w}x{scaled_h} ({scaled_pixels:,} pixels)")
        
        # Check if scaling was needed
        if original_pixels > max_pixels:
            reduction = (1 - scaled_pixels / original_pixels) * 100
            print(f"Reduced by: {reduction:.1f}%")
        else:
            print("No scaling needed")
        
        # Verify aspect ratio is maintained (within rounding tolerance)
        original_ratio = original_w / original_h
        scaled_ratio = scaled_w / scaled_h
        ratio_diff = abs(original_ratio - scaled_ratio) / original_ratio * 100
        print(f"Aspect ratio preserved: {ratio_diff:.2f}% difference")
        
        print("-" * 40)

if __name__ == "__main__":
    test_scaling()

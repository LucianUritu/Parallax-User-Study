import cv2
import numpy as np
import asyncio
import websockets
import json
import threading

class FastInteractiveParallax:
    def __init__(self):
        # Load images
        self.original_img = cv2.imread("image.jpg")
        self.parallax_img = cv2.imread("parallax_multidirectional.png")
        self.depth_img = cv2.imread("depth.png", 0)
        
        if self.original_img is None or self.parallax_img is None or self.depth_img is None:
            print("Error: Could not load required images")
            return
            
        # Resize for better performance if images are large
        max_width = 600  # Reduced for better performance
        if self.original_img.shape[1] > max_width:
            scale = max_width / self.original_img.shape[1]
            new_width = int(self.original_img.shape[1] * scale)
            new_height = int(self.original_img.shape[0] * scale)
            self.original_img = cv2.resize(self.original_img, (new_width, new_height))
            self.parallax_img = cv2.resize(self.parallax_img, (new_width, new_height))
            self.depth_img = cv2.resize(self.depth_img, (new_width, new_height))
        
        # Get image dimensions
        self.h, self.w = self.original_img.shape[:2]
        
        # Mouse position (normalized to -1 to 1)
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        
        # Eye tracking variables
        self.eye_x = 0.0
        self.eye_y = 0.0
        self.use_eye_tracking = False

        # Maximum shift amount
        self.max_shift = 20
        
        # Pre-compute depth normalization for performance
        self.depth_norm = self.depth_img.astype(np.float32) / 255.0
        
        # Sweet spot for focus (can be randomized)
        self.sweet_spot_x = np.random.uniform(-0.5, 0.5)
        self.sweet_spot_y = np.random.uniform(-0.5, 0.5)
        self.focus_radius = 0.3
        
        # Create coordinate grids for efficient computation
        self.y_grid, self.x_grid = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        
        print(f"Loaded images: {self.w}x{self.h}")
        print(f"Sweet spot: ({self.sweet_spot_x:.2f}, {self.sweet_spot_y:.2f})")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse movement"""
        # Normalize mouse position to -1 to 1
        self.mouse_x = (x / self.w) * 2.0 - 1.0
        self.mouse_y = (y / self.h) * 2.0 - 1.0

    async def receive_eye_data(self):
        """Receive eye tracking data from WebSocket server"""
        try:
            async with websockets.connect("ws://localhost:8765") as websocket:
                self.use_eye_tracking = True
                print("Connected to eye tracking system")
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    self.eye_x = data.get("x", 0.0)
                    self.eye_y = data.get("y", 0.0)
        except Exception as e:
            print("Eye tracking connection failed:", e)
            self.use_eye_tracking = False
        
    def calculate_focus_quality(self):
        """Calculate how close we are to the sweet spot (0=blurry, 1=sharp)"""
         # Use eye tracking if available, otherwise use mouse
        if self.use_eye_tracking:
            current_x = self.eye_x
            current_y = self.eye_y
        else:
            current_x = self.mouse_x
            current_y = self.mouse_y
            
        distance_x = abs(current_x - self.sweet_spot_x)
        distance_y = abs(current_y - self.sweet_spot_y)
        distance = np.sqrt(distance_x**2 + distance_y**2)
        
        # Convert distance to focus quality (closer = better focus)
        focus_quality = max(0.0, 1.0 - (distance / self.focus_radius))
        return focus_quality
        
    def create_parallax_fast(self):
        """Create optimized parallax effect with depth-based shifting"""
        focus_quality = self.calculate_focus_quality()
        
        # Use eye tracking if available, otherwise use mouse
        if self.use_eye_tracking:
            current_x = self.eye_x
            current_y = self.eye_y
        else:
            current_x = self.mouse_x
            current_y = self.mouse_y
        
        # Calculate depth-based shifts using vectorized operations
        # Foreground (depth < 0.5) shifts one way, background shifts opposite
        foreground_mask = self.depth_norm < 0.5
        background_mask = ~foreground_mask
        
        # Calculate shift amounts based on current position
        shift_x = current_x * self.max_shift
        shift_y = current_y * self.max_shift
        
        # Create shift maps
        shift_map_x = np.zeros_like(self.depth_norm)
        shift_map_y = np.zeros_like(self.depth_norm)
        
        # Foreground shifts (depth 0-0.5 maps to 0-1 multiplier)
        shift_map_x[foreground_mask] = shift_x * (self.depth_norm[foreground_mask] * 2)
        shift_map_y[foreground_mask] = shift_y * (self.depth_norm[foreground_mask] * 2)
        
        # Background shifts (depth 0.5-1 maps to 0-1 multiplier, but negative)
        shift_map_x[background_mask] = -shift_x * ((self.depth_norm[background_mask] - 0.5) * 2)
        shift_map_y[background_mask] = -shift_y * ((self.depth_norm[background_mask] - 0.5) * 2)
        
        # Create new coordinate maps
        new_x = self.x_grid + shift_map_x
        new_y = self.y_grid + shift_map_y
        
        # Apply transformation using cv2.remap for efficiency
        output = cv2.remap(self.original_img, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Apply focus-dependent blur
        if focus_quality < 1.0:
            blur_amount = int((1.0 - focus_quality) * 8) + 1
            if blur_amount % 2 == 0:
                blur_amount += 1  # Ensure odd number
            if blur_amount > 1:
                output = cv2.GaussianBlur(output, (blur_amount, blur_amount), 0)
        
        return output, focus_quality
        
    def run(self):
        """Run the interactive parallax viewer"""
        print("Fast Interactive Parallax Viewer (Using parallax.png)")
        print("Move your mouse to control the parallax effect!")
        print("Starting eye tracking connection...")
        print("Mouse X: Controls blend between original and parallax")
        print("Mouse Y: Controls vertical shift for both images")
        print("Keys:")
        print("  R - Reset to center")
        print("  + - Increase max shift")
        print("  - - Decrease max shift")
        print("  S - Save current view")
        print("  Q - Quit")
        
        # Start eye tracking in a separate thread
        def start_eye_tracking():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.receive_eye_data())
            
        eye_thread = threading.Thread(target=start_eye_tracking, daemon=True)
        eye_thread.start()
        
        cv2.namedWindow('Fast Interactive Parallax', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Fast Interactive Parallax', self.mouse_callback)
        
        while True:
            # Create parallax effect
            parallax_img, focus_quality = self.create_parallax_fast()
            
            # Add text overlay showing current settings
            color = (0, 255, 0) if focus_quality > 0.8 else (0, 255, 255) if focus_quality > 0.5 else (0, 0, 255)
            
            # Use eye tracking if available, otherwise use mouse
            if self.use_eye_tracking:
                current_x = self.eye_x
                current_y = self.eye_y
                input_source = "Eye"
            else:
                current_x = self.mouse_x
                current_y = self.mouse_y
                input_source = "Mouse"
            
            info_text = f"Focus: {focus_quality:.2f} | {input_source}: ({current_x:.2f}, {current_y:.2f}) | Max: {self.max_shift}"

            cv2.putText(parallax_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if focus_quality > 0.9:
                cv2.putText(parallax_img, "PERFECT FOCUS!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif focus_quality > 0.7:
                cv2.putText(parallax_img, "Getting closer...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show current parallax image
            cv2.imshow('Fast Interactive Parallax', parallax_img)
            
            # Handle key presses
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset to center
                self.mouse_x = 0.0
                self.mouse_y = 0.0
            elif key == ord('+') or key == ord('='):
                # Increase max shift
                self.max_shift = min(200, self.max_shift + 5)
                print(f"Max shift: {self.max_shift}")
            elif key == ord('-'):
                # Decrease max shift
                self.max_shift = max(5, self.max_shift - 5)
                print(f"Max shift: {self.max_shift}")
            elif key == ord('s'):
                # Save current view
                filename = f"parallax_manual_{self.mouse_x:.2f}_{self.mouse_y:.2f}.png"
                cv2.imwrite(filename, parallax_img)
                print(f"Saved: {filename}")
        
        cv2.destroyAllWindows()

class SimpleBlendParallax:
    """Simple version that blends between original and pre-made parallax"""
    def __init__(self):
        self.original_img = cv2.imread("image.jpg")
        self.parallax_img = cv2.imread("parallax.png")
        
        if self.original_img is None or self.parallax_img is None:
            print("Error: Could not load required images")
            return
            
        # Resize for consistency
        max_width = 800
        if self.original_img.shape[1] > max_width:
            scale = max_width / self.original_img.shape[1]
            new_width = int(self.original_img.shape[1] * scale)
            new_height = int(self.original_img.shape[0] * scale)
            self.original_img = cv2.resize(self.original_img, (new_width, new_height))
            self.parallax_img = cv2.resize(self.parallax_img, (new_width, new_height))
            
        self.h, self.w = self.original_img.shape[:2]
        self.mouse_x = 0.0
        
        print(f"Loaded images: {self.w}x{self.h}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse movement - only use X axis"""
        self.mouse_x = (x / self.w) * 2.0 - 1.0
        
    def run(self):
        """Run simple interactive viewer"""
        print("Simple Interactive Parallax")
        print("Move mouse left/right to blend between original and parallax")
        print("Press Q to quit")
        
        cv2.namedWindow('Simple Parallax Blend', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Simple Parallax Blend', self.mouse_callback)
        
        while True:
            # Blend between original and parallax based on mouse X position
            blend_factor = (self.mouse_x + 1.0) / 2.0
            blend_factor = max(0.0, min(1.0, blend_factor))  # Clamp to 0-1
            
            # Create blended image
            blended = cv2.addWeighted(
                self.original_img, 1.0 - blend_factor,
                self.parallax_img, blend_factor,
                0
            )
            
            # Add text overlay
            info_text = f"Blend: {blend_factor:.2f} | Original <-- --> Parallax"
            cv2.putText(blended, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Simple Parallax Blend', blended)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
                
        cv2.destroyAllWindows()

def main():
    print("Choose interactive parallax mode:")
    print("1. Fast interactive (mouse controls image shift)")
    print("2. Simple blend (mouse blends original/parallax)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        viewer = FastInteractiveParallax()
        if viewer.original_img is not None:
            viewer.run()
    elif choice == "2":
        viewer = SimpleBlendParallax()
        if viewer.original_img is not None:
            viewer.run()
    else:
        print("Invalid choice. Starting simple blend...")
        viewer = SimpleBlendParallax()
        if viewer.original_img is not None:
            viewer.run()

if __name__ == "__main__":
    main()
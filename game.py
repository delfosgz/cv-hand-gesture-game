import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from collections import deque
import random
import time

# Screen resolution 
def get_screen_resolution():
    try:
        root = tk.Tk(); root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy(); return sw, sh
    except tk.TclError:
        print("Advertencia: Tkinter falló. Usando 1920x1080 por defecto."); return 1920, 1080

SCREEN_W, SCREEN_H = get_screen_resolution()

# MediaPipe Hands Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Cámara no accesible."); exit()
ret, frame_test = cap.read()
if not ret: print("Error: No se pudo leer frame."); cap.release(); exit()

PROCESSING_HEIGHT, PROCESSING_WIDTH, _ = frame_test.shape
print(f"Resolución de cámara detectada (usada para procesamiento principal): {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")

MP_INPUT_PROCESSING_WIDTH = 480
MP_ASPECT_RATIO = float(PROCESSING_HEIGHT) / PROCESSING_WIDTH
MP_INPUT_PROCESSING_HEIGHT = int(MP_INPUT_PROCESSING_WIDTH * MP_ASPECT_RATIO)
print(f"MediaPipe procesará a: {MP_INPUT_PROCESSING_WIDTH}x{MP_INPUT_PROCESSING_HEIGHT}")

TARGET_DISPLAY_WIDTH = 700 # Target width for the display windows
DISPLAY_ASPECT_RATIO = float(PROCESSING_HEIGHT) / PROCESSING_WIDTH
DISPLAY_W, DISPLAY_H = TARGET_DISPLAY_WIDTH, int(TARGET_DISPLAY_WIDTH * DISPLAY_ASPECT_RATIO)

# General
SKY_BLUE_BACKGROUND = (230, 216, 173)  # BGR: Light Sky Blue for game canvas
TEXT_SHADOW_COLOR = (80, 80, 80)       # Darker gray for better contrast
TEXT_PROBLEM_COLOR = (50, 50, 50)      # Dark gray for math problem 
TEXT_ANSWER_COLOR = (70, 70, 70)       # Dark gray for 
FEEDBACK_CORRECT_COLOR = (0, 160, 0)   # Vibrant Green 
FEEDBACK_INCORRECT_COLOR = (0, 0, 200) # Bright Red 

# Hand Landmark Colors (MediaPipe drawing)
LANDMARK_POINT_COLOR = (255, 100, 0)   # Vibrant Blue 
LANDMARK_CONNECTION_COLOR = (0, 200, 255) # Sunny Yellow/Gold for connections

# Particle Colors
SPARKLE_COLORS_BGR = [
    (0, 255, 255),    # Bright Yellow
    (255, 192, 203),  # Pink
    (144, 238, 144),  # Light Green
    (0, 165, 255),    # Orange
    (255, 255, 255),  # White
    (220, 220, 220),  # Light Gray
    (255, 105, 180),  # Hot Pink
    (135, 206, 250),  # Light Sky Blue (different from background)
    (255, 175, 0)     # Light Blue / Cyan
]

# Base Colors 
GEM_HAND_SOLID_COLOR_BGR = {
    "ruby": (0, 0, 220),          # Ruby Red
    "emerald": (0, 200, 0),      # Emerald Green
    "sapphire": (220, 0, 0),     # Sapphire Blue
    "amethyst": (180, 0, 128),   # Amethyst Purple
    "citrine": (0, 220, 220),    # Citrine Yellow
    "diamond": (230, 230, 230)   # Bright Diamond-like
}

# "Alien Hand" Base Color
ALIEN_BASE_COLOR_BGR = (100, 180, 100) # Muted green for alien hand

# "Magic Aura" Effect
AURA_COLOR_ON_BLUE_BG = (200, 200, 255) # Soft, glowing light pink/lavender aura

# "Speedy Fingers" Trail Effect
TRAIL_COLOR_ON_BLUE_BG = (220, 235, 245) # Ethereal light blue/white, slightly different from sky

# --- Game Theming: Visual Effects Flags ---
RAINBOW_HAND_MODE = False
MAGIC_AURA_MODE = True
SPEEDY_FINGERS_MODE = False

# --- Hand Rendering Parameters ---
BASE_PARTICLE_MAX_RADIUS = 12
NUM_INTERPOLATED_POINTS_PER_SEGMENT = 2
MIN_DEPTH_SCALE = 0.5; MAX_DEPTH_SCALE = 1.5

AURA_DILATION_KERNEL_SIZE = (13, 13)
GHOST_TRAIL_LENGTH = 4

# Options: "sparkle", "ruby", "emerald", "sapphire", "amethyst", "citrine", "diamond", "alien"
HAND_COLOR_MODE = "alien" 


ghost_masks = deque(maxlen=GHOST_TRAIL_LENGTH)

# --- Game State & Math Problem Logic ---
game_state = "PREGUNTANDO"; math_problem_str = ""; current_correct_answer = 0
feedback_message = ""; feedback_message_color = FEEDBACK_CORRECT_COLOR; feedback_timer_start = 0
FEEDBACK_DURATION_SECONDS = 2.0; player_answered_this_round = False

stable_finger_count = 0; candidate_finger_count = 0; candidate_stability_counter = 0
STABILITY_THRESHOLD_FRAMES = 7

# --- Window Theming ---
GAME_SCREEN_NAME = 'Finger Count Game'
HELPER_WINDOW_NAME = '(Cámara)'

quad_w, quad_h = SCREEN_W // 2, SCREEN_H // 2
debug_x = quad_w + (quad_w - DISPLAY_W) // 2; debug_y = (quad_h - DISPLAY_H) // 2
game_x = quad_w + (quad_w - DISPLAY_W) // 2; game_y = quad_h + (quad_h - DISPLAY_H) // 2
debug_x = max(0, debug_x); debug_y = max(0, debug_y)
game_x = max(0, game_x); game_y = max(quad_h, game_y)

if SCREEN_W < DISPLAY_W * 2:
    debug_x = (SCREEN_W - DISPLAY_W) // 2
    game_x = (SCREEN_W - DISPLAY_W) // 2
    if SCREEN_H < DISPLAY_H * 2:
         debug_y = 0
         game_y = DISPLAY_H + 10
    else:
        debug_y = (SCREEN_H - DISPLAY_H) // 2
        game_y = (SCREEN_H - DISPLAY_H) // 2


window_configs = [
    (HELPER_WINDOW_NAME, DISPLAY_W, DISPLAY_H, debug_x, debug_y),
    (GAME_SCREEN_NAME, DISPLAY_W, DISPLAY_H, game_x, game_y)
]
for name, w, h, x, y in window_configs:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL); cv2.resizeWindow(name, w, h); cv2.moveWindow(name, x, y)

hue_shift_value = 0; aura_kernel = np.ones(AURA_DILATION_KERNEL_SIZE, np.uint8)


finger_tip_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
finger_pip_ids = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
finger_mcp_ids = [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]

landmark_to_id_map = {lm_idx: lm_id for lm_id_list in [finger_tip_ids, finger_pip_ids, finger_mcp_ids] for lm_idx, lm_id in enumerate(lm_id_list)}


def interpolate_points_3d(p1, p2, n):
    pts = []
    for i in range(1, n + 1):
        t = i / (n + 1.0)
        pt_x = int(p1[0] * (1 - t) + p2[0] * t)
        pt_y = int(p1[1] * (1 - t) + p2[1] * t)
        pt_z = p1[2] * (1 - t) + p2[2] * t
        pts.append((pt_x, pt_y, pt_z))
    return pts

def generate_math_problem():
    global math_problem_str, current_correct_answer, player_answered_this_round
    player_answered_this_round = False
    while True:
        num_terms = 2
        terms = [random.randint(1, 9) for _ in range(num_terms)]
        ops_choices = ['+', '-']
        current_op = random.choice(ops_choices)
        
        if current_op == '-' and terms[0] < terms[1]:
            terms[0], terms[1] = terms[1], terms[0]
            
        problem_display_str = f"{terms[0]} {current_op} {terms[1]}"
        calculated_result = eval(f"{terms[0]} {current_op} {terms[1]}")

        if 1 <= calculated_result <= 5:
            math_problem_str = problem_display_str + " = ?"
            current_correct_answer = int(calculated_result)
            print(f"Nuevo problema: {math_problem_str} (Respuesta: {current_correct_answer})")
            return

generate_math_problem()
print(f"Modo Mano Arcoiris: {'Activado' if RAINBOW_HAND_MODE else 'Desactivado'}")
print(f"Modo Aura Mágica: {'Activado' if MAGIC_AURA_MODE else 'Desactivado'}")
print(f"Modo Dedos Veloces: {'Activado' if SPEEDY_FINGERS_MODE else 'Desactivado'}")
print(f"Modo Color de Mano: {HAND_COLOR_MODE}")
print("¡Bienvenido el juego de contar dedos!")
print("Muestra la respuesta con tus dedos. Presiona 'q' para salir.")

# --- Main Game Loop ---
while cap.isOpened():
    success, image_cam_orig = cap.read()
    if not success: print("Frame vacío."); continue

    image_cam_flipped = cv2.flip(image_cam_orig, 1)
    debug_image_processed = image_cam_flipped.copy()

    image_for_mp = cv2.resize(image_cam_flipped, (MP_INPUT_PROCESSING_WIDTH, MP_INPUT_PROCESSING_HEIGHT))
    image_rgb_for_mp = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
    image_rgb_for_mp.flags.writeable = False
    results = hands.process(image_rgb_for_mp)
    image_rgb_for_mp.flags.writeable = True

    game_canvas = np.full((PROCESSING_HEIGHT, PROCESSING_WIDTH, 3), SKY_BLUE_BACKGROUND, dtype=np.uint8)
    current_hand_mask = np.zeros((PROCESSING_HEIGHT, PROCESSING_WIDTH), dtype=np.uint8)
    current_raw_finger_count = 0
    hand_detected_this_frame = False
    
    fingertip_coords_set = set()


    if results.multi_hand_landmarks and results.multi_handedness:
        hand_detected_this_frame = True
        for hand_idx, hand_landmarks_mp in enumerate(results.multi_hand_landmarks):
            handedness_label = results.multi_handedness[hand_idx].classification[0].label
            
            mp_drawing.draw_landmarks(
                debug_image_processed, hand_landmarks_mp, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=LANDMARK_POINT_COLOR, thickness=3, circle_radius=5),
                mp_drawing.DrawingSpec(color=LANDMARK_CONNECTION_COLOR, thickness=3, circle_radius=3)
            )
            
            landmarks_list = hand_landmarks_mp.landmark
            # --- Finger Counting Logic (Maintained) ---
            fingers_up = [False] * 5
            thumb_tip = landmarks_list[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = landmarks_list[mp_hands.HandLandmark.THUMB_IP]
            thumb_mcp = landmarks_list[mp_hands.HandLandmark.THUMB_MCP]
            index_finger_pip = landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_finger_mcp = landmarks_list[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_finger_pip = landmarks_list[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            
            Y_THRESH_THUMB_TIP_VS_IP = 0.03
            Y_THRESH_THUMB_TIP_VS_INDEX_MCP = 0.02
            thumb_vertically_extended = (thumb_tip.y < thumb_ip.y - Y_THRESH_THUMB_TIP_VS_IP) and \
                                      (thumb_tip.y < index_finger_mcp.y - Y_THRESH_THUMB_TIP_VS_INDEX_MCP)
            X_THRESH_THUMB_TIP_VS_IP = 0.025
            X_THRESH_THUMB_IP_VS_MCP = 0.020
            thumb_horizontally_extended = False
            if handedness_label == "Right":
                thumb_horizontally_extended = (thumb_tip.x < thumb_ip.x - X_THRESH_THUMB_TIP_VS_IP) and \
                                              (thumb_ip.x < thumb_mcp.x - X_THRESH_THUMB_IP_VS_MCP)
            elif handedness_label == "Left":
                thumb_horizontally_extended = (thumb_tip.x > thumb_ip.x + X_THRESH_THUMB_TIP_VS_IP) and \
                                              (thumb_ip.x > thumb_mcp.x + X_THRESH_THUMB_IP_VS_MCP)
            
            is_thumb_potentially_up = thumb_vertically_extended or thumb_horizontally_extended
            
            ref_dist = np.hypot(index_finger_pip.x - index_finger_mcp.x, 
                                index_finger_pip.y - index_finger_mcp.y) + 1e-6
            dist_thumb_tip_to_idx_pip = np.hypot(thumb_tip.x - index_finger_pip.x, thumb_tip.y - index_finger_pip.y)
            dist_thumb_tip_to_mid_pip = np.hypot(thumb_tip.x - middle_finger_pip.x, thumb_tip.y - middle_finger_pip.y)
            
            PROXIMITY_SCALE_FACTOR = 0.9 
            thumb_close_to_fingers = (dist_thumb_tip_to_idx_pip < ref_dist * PROXIMITY_SCALE_FACTOR) or \
                                     (dist_thumb_tip_to_mid_pip < ref_dist * PROXIMITY_SCALE_FACTOR)

            NOT_CLEARLY_ABOVE_SCALE_FACTOR = 0.3
            thumb_not_clearly_above_closed_fingers = (thumb_tip.y > index_finger_pip.y - ref_dist * NOT_CLEARLY_ABOVE_SCALE_FACTOR)

            TUCKED_IN_MARGIN_SCALE = 0.2
            Y_OFFSET_FROM_THUMB_MCP = 0.02 
            thumb_tucked_in = False
            if handedness_label == "Right":
                if (thumb_tip.x > index_finger_mcp.x - ref_dist * TUCKED_IN_MARGIN_SCALE) and \
                   (thumb_tip.y > thumb_mcp.y - Y_OFFSET_FROM_THUMB_MCP):
                    thumb_tucked_in = True
            elif handedness_label == "Left":
                 if (thumb_tip.x < index_finger_mcp.x + ref_dist * TUCKED_IN_MARGIN_SCALE) and \
                    (thumb_tip.y > thumb_mcp.y - Y_OFFSET_FROM_THUMB_MCP):
                    thumb_tucked_in = True

            if is_thumb_potentially_up:
                override_as_closed = False
                if thumb_close_to_fingers and thumb_not_clearly_above_closed_fingers:
                    override_as_closed = True
                elif thumb_tucked_in and not thumb_vertically_extended : 
                    override_as_closed = True
                fingers_up[0] = not override_as_closed
            else:
                fingers_up[0] = False

            Y_THRESH_OTHER_FINGERS_TIP_VS_PIP = 0.045 
            for i in range(1, 5): 
                tip = landmarks_list[finger_tip_ids[i]]
                pip = landmarks_list[finger_pip_ids[i]]
                if tip.y < pip.y - Y_THRESH_OTHER_FINGERS_TIP_VS_PIP:
                    fingers_up[i] = True
                else:
                    fingers_up[i] = False
            current_raw_finger_count = sum(fingers_up)
            # --- End Finger Counting Logic ---

            landmarks_3d, all_z = [], []
            # Store direct fingertip landmark coordinates for the "alien" effect
            direct_fingertip_coords = []
            for tip_id_enum in finger_tip_ids:
                lm = landmarks_list[tip_id_enum]
                direct_fingertip_coords.append((int(lm.x * PROCESSING_WIDTH), int(lm.y * PROCESSING_HEIGHT)))

            for lm_idx, lm in enumerate(landmarks_list): 
                px,py,pz = int(lm.x*PROCESSING_WIDTH), int(lm.y*PROCESSING_HEIGHT), lm.z
                landmarks_3d.append((px,py,pz)) # This list stores (x,y,z) tuples for all 21 landmarks
                all_z.append(pz)
            
            if not landmarks_3d: continue
            min_z,max_z = min(all_z),max(all_z)
            delta_z = max_z - min_z if (max_z - min_z) > 1e-6 else 1e-6

            all_pts_3d = set() # Using a set to avoid duplicate points
            # Add original landmarks to all_pts_3d
            for lm_data in landmarks_3d: # lm_data is (px,py,pz)
                all_pts_3d.add(lm_data)

            if mp_hands.HAND_CONNECTIONS:
                for conn in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = conn
                    if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
                        p1, p2 = landmarks_3d[start_idx], landmarks_3d[end_idx]
                        interpolated = interpolate_points_3d(p1, p2, NUM_INTERPOLATED_POINTS_PER_SEGMENT)
                        for p_i in interpolated: all_pts_3d.add(p_i)
            
            # Create current_hand_mask using circles for aura and rainbow effect
            for px, py, pz in all_pts_3d:
                norm_depth = (pz - min_z) / delta_z
                scale_factor = MAX_DEPTH_SCALE - (norm_depth * (MAX_DEPTH_SCALE - MIN_DEPTH_SCALE))
                particle_radius = max(1, int(BASE_PARTICLE_MAX_RADIUS * scale_factor))
                
                center_x = max(particle_radius, min(px, PROCESSING_WIDTH - particle_radius - 1))
                center_y = max(particle_radius, min(py, PROCESSING_HEIGHT - particle_radius - 1))
                if center_x + particle_radius <= PROCESSING_WIDTH and \
                   center_y + particle_radius <= PROCESSING_HEIGHT and \
                   center_x - particle_radius >= 0 and \
                   center_y - particle_radius >= 0:
                    cv2.circle(current_hand_mask, (center_x, center_y), particle_radius, 255, -1)

            if MAGIC_AURA_MODE and current_hand_mask.any():
                aura_mask_dilated = cv2.dilate(current_hand_mask, aura_kernel, iterations=1)
                aura_only_mask = cv2.subtract(aura_mask_dilated, current_hand_mask)
                game_canvas[aura_only_mask == 255] = AURA_COLOR_ON_BLUE_BG
            
            if RAINBOW_HAND_MODE and current_hand_mask.any():
                hsv_canvas = cv2.cvtColor(game_canvas, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_canvas)
                hand_hsv_layer = np.zeros_like(game_canvas, dtype=np.uint8)
                hand_hsv_layer[:,:,0] = (hue_shift_value % 180)
                hand_hsv_layer[:,:,1] = 230
                hand_hsv_layer[:,:,2] = 230
                hand_bgr_layer = cv2.cvtColor(hand_hsv_layer, cv2.COLOR_HSV2BGR)
                game_canvas = np.where(cv2.cvtColor(current_hand_mask, cv2.COLOR_GRAY2BGR) == 255, hand_bgr_layer, game_canvas)

            elif HAND_COLOR_MODE == "sparkle":
                for px, py, pz in all_pts_3d:
                    norm_depth = (pz - min_z) / delta_z
                    scale_factor = MAX_DEPTH_SCALE - (norm_depth * (MAX_DEPTH_SCALE - MIN_DEPTH_SCALE))
                    particle_radius = max(1, int(BASE_PARTICLE_MAX_RADIUS * scale_factor))
                    center_x = max(particle_radius, min(px, PROCESSING_WIDTH - particle_radius - 1))
                    center_y = max(particle_radius, min(py, PROCESSING_HEIGHT - particle_radius - 1))
                    sparkle_color = random.choice(SPARKLE_COLORS_BGR)
                    if center_x + particle_radius <= PROCESSING_WIDTH and \
                       center_y + particle_radius <= PROCESSING_HEIGHT and \
                       center_x - particle_radius >= 0 and \
                       center_y - particle_radius >= 0:
                        cv2.circle(game_canvas, (center_x, center_y), particle_radius, sparkle_color, -1)
            
            elif HAND_COLOR_MODE in GEM_HAND_SOLID_COLOR_BGR:
                base_gem_color = GEM_HAND_SOLID_COLOR_BGR[HAND_COLOR_MODE]
                for px, py, pz in all_pts_3d:
                    norm_depth = (pz - min_z) / delta_z
                    scale_factor = MAX_DEPTH_SCALE - (norm_depth * (MAX_DEPTH_SCALE - MIN_DEPTH_SCALE))
                    particle_radius = max(1, int(BASE_PARTICLE_MAX_RADIUS * scale_factor))
                    center_x = max(particle_radius, min(px, PROCESSING_WIDTH - particle_radius - 1))
                    center_y = max(particle_radius, min(py, PROCESSING_HEIGHT - particle_radius - 1))
                    flicker = random.uniform(0.8, 1.2)
                    current_gem_color = tuple(np.clip(c * flicker, 0, 255).astype(np.uint8) for c in base_gem_color)
                    if center_x + particle_radius <= PROCESSING_WIDTH and \
                       center_y + particle_radius <= PROCESSING_HEIGHT and \
                       center_x - particle_radius >= 0 and \
                       center_y - particle_radius >= 0:
                        cv2.circle(game_canvas, (center_x, center_y), particle_radius, current_gem_color, -1)
            
            elif HAND_COLOR_MODE == "alien":
                # Check if a point is a fingertip by comparing its original landmark index
                # This requires knowing the original MediaPipe index for each point in all_pts_3d
                # For simplicity, we'll check proximity to the direct_fingertip_coords
                fingertip_proximity_threshold = BASE_PARTICLE_MAX_RADIUS * 1.5 # How close a particle needs to be to a tip

                for px, py, pz in all_pts_3d:
                    norm_depth = (pz - min_z) / delta_z
                    scale_factor = MAX_DEPTH_SCALE - (norm_depth * (MAX_DEPTH_SCALE - MIN_DEPTH_SCALE))
                    current_particle_radius = max(1, int(BASE_PARTICLE_MAX_RADIUS * scale_factor))
                    
                    is_fingertip_particle = False
                    for tip_x, tip_y in direct_fingertip_coords:
                        dist_sq = (px - tip_x)**2 + (py - tip_y)**2
                        if dist_sq < fingertip_proximity_threshold**2 :
                            is_fingertip_particle = True
                            break
                    
                    if is_fingertip_particle:
                        current_particle_radius = max(1, int(current_particle_radius * 1.3)) # Make fingertips slightly larger

                    center_x = max(current_particle_radius, min(px, PROCESSING_WIDTH - current_particle_radius - 1))
                    center_y = max(current_particle_radius, min(py, PROCESSING_HEIGHT - current_particle_radius - 1))
                    
                    # Mottled color for alien skin
                    mottled_color = [
                        np.clip(c * random.uniform(0.85, 1.15), 0, 255) for c in ALIEN_BASE_COLOR_BGR
                    ]
                    alien_particle_color = tuple(int(c) for c in mottled_color)

                    if center_x + current_particle_radius <= PROCESSING_WIDTH and \
                       center_y + current_particle_radius <= PROCESSING_HEIGHT and \
                       center_x - current_particle_radius >= 0 and \
                       center_y - current_particle_radius >= 0:
                        cv2.circle(game_canvas, (center_x, center_y), current_particle_radius, alien_particle_color, -1)
            break 
            
    if current_raw_finger_count == candidate_finger_count: candidate_stability_counter += 1
    else: candidate_finger_count = current_raw_finger_count; candidate_stability_counter = 1
    
    if candidate_stability_counter >= STABILITY_THRESHOLD_FRAMES:
        if stable_finger_count != candidate_finger_count:
            stable_finger_count = candidate_finger_count
            if game_state == "PREGUNTANDO" and hand_detected_this_frame: 
                player_answered_this_round = False
    
    if game_state == "PREGUNTANDO":
        if not player_answered_this_round and 1 <= stable_finger_count <= 5 and hand_detected_this_frame:
            player_answered_this_round = True 
            if stable_finger_count == current_correct_answer:
                feedback_message = "MUY BIEN"; feedback_message_color = FEEDBACK_CORRECT_COLOR
            else:
                feedback_message = "INTENTA DE NUEVO"; feedback_message_color = FEEDBACK_INCORRECT_COLOR
            game_state = "FEEDBACK"; feedback_timer_start = time.time()
    elif game_state == "FEEDBACK":
        if time.time() - feedback_timer_start > FEEDBACK_DURATION_SECONDS:
            generate_math_problem(); game_state = "PREGUNTANDO"; feedback_message = ""
            stable_finger_count = 0 
            candidate_finger_count = 0
            candidate_stability_counter = 0

    font_game = cv2.FONT_HERSHEY_DUPLEX
    
    if game_state=="PREGUNTANDO":
        (w,h),_ = cv2.getTextSize(math_problem_str, font_game, 1.5, 3)
        cv2.putText(game_canvas,math_problem_str,((PROCESSING_WIDTH-w)//2 + 3, 80 + 3),font_game,1.5,TEXT_SHADOW_COLOR,3,cv2.LINE_AA)
        cv2.putText(game_canvas,math_problem_str,((PROCESSING_WIDTH-w)//2, 80),font_game,1.5,TEXT_PROBLEM_COLOR,3,cv2.LINE_AA)

    if feedback_message:
        font_scale_feedback = 1.8
        text_to_display = feedback_message
        
        if feedback_message == "MUY BIEN":
            elapsed_feedback_time = time.time() - feedback_timer_start
            pulse_frequency = 2
            font_scale_feedback = 1.8 * (1 + 0.1 * np.sin(elapsed_feedback_time * pulse_frequency * np.pi * 2 / FEEDBACK_DURATION_SECONDS))
            if int(elapsed_feedback_time * 5) % 2 == 0 :
                 text_to_display = f"* {feedback_message} *"

        offset_x, offset_y = 0, 0
        if feedback_message=="INTENTA DE NUEVO" and int(time.time()*10)%2==0:
            offset_x=random.randint(-5,5);offset_y=random.randint(-5,5)
        
        (w,h),_ = cv2.getTextSize(text_to_display, font_game, font_scale_feedback, 3)
        cv2.putText(game_canvas,text_to_display,((PROCESSING_WIDTH-w)//2+offset_x + 3,PROCESSING_HEIGHT//2+offset_y + 3 + h//2),font_game,font_scale_feedback,TEXT_SHADOW_COLOR,3,cv2.LINE_AA)
        cv2.putText(game_canvas,text_to_display,((PROCESSING_WIDTH-w)//2+offset_x,PROCESSING_HEIGHT//2+offset_y + h//2),font_game,font_scale_feedback,feedback_message_color,3,cv2.LINE_AA)

    player_answer_txt = f"Dedos: {stable_finger_count if stable_finger_count > 0 and hand_detected_this_frame else '-'}"
    (w_r,h_r),_ = cv2.getTextSize(player_answer_txt, font_game, 1.2, 2)
    cv2.putText(game_canvas,player_answer_txt,((PROCESSING_WIDTH-w_r)//2 + 2, PROCESSING_HEIGHT-40 + 2),font_game,1.2,TEXT_SHADOW_COLOR,2,cv2.LINE_AA)
    cv2.putText(game_canvas,player_answer_txt,((PROCESSING_WIDTH-w_r)//2, PROCESSING_HEIGHT-40),font_game,1.2,TEXT_ANSWER_COLOR,2,cv2.LINE_AA)

    if SPEEDY_FINGERS_MODE and current_hand_mask.any():
        mask_for_trail = np.zeros_like(current_hand_mask)
        if hand_detected_this_frame and results.multi_hand_landmarks:
             # Re-populate all_pts_3d if not already available or if it needs to be specific to trail
            temp_all_pts_3d_trail = set()
            # Assuming landmarks_3d and direct_fingertip_coords are from the current hand detection
            for lm_data in landmarks_3d: # lm_data is (px,py,pz)
                temp_all_pts_3d_trail.add(lm_data)
            if mp_hands.HAND_CONNECTIONS:
                for conn in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = conn
                    if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
                        p1_trail, p2_trail = landmarks_3d[start_idx], landmarks_3d[end_idx]
                        interpolated_trail = interpolate_points_3d(p1_trail, p2_trail, NUM_INTERPOLATED_POINTS_PER_SEGMENT)
                        for p_i_trail in interpolated_trail: temp_all_pts_3d_trail.add(p_i_trail)
            
            min_z_trail, max_z_trail = min(all_z) if all_z else 0, max(all_z) if all_z else 1 # Use all_z from main processing
            delta_z_trail = max_z_trail - min_z_trail if (max_z_trail - min_z_trail) > 1e-6 else 1e-6


            for px, py, pz in temp_all_pts_3d_trail:
                norm_depth = (pz - min_z_trail) / delta_z_trail if delta_z_trail > 1e-6 else 0.5
                scale_factor = MAX_DEPTH_SCALE - (norm_depth * (MAX_DEPTH_SCALE - MIN_DEPTH_SCALE))
                particle_radius = max(1, int(BASE_PARTICLE_MAX_RADIUS * scale_factor * 0.8))
                center_x = max(particle_radius, min(px, PROCESSING_WIDTH - particle_radius - 1))
                center_y = max(particle_radius, min(py, PROCESSING_HEIGHT - particle_radius - 1))
                if center_x + particle_radius <= PROCESSING_WIDTH and \
                   center_y + particle_radius <= PROCESSING_HEIGHT and \
                   center_x - particle_radius >= 0 and \
                   center_y - particle_radius >= 0:
                    cv2.circle(mask_for_trail, (center_x, center_y), particle_radius, 255, -1)
        
        if mask_for_trail.any():
            ghost_masks.append(mask_for_trail)

        for i, gm_item in enumerate(ghost_masks):
            if gm_item.any():
                opacity = (i + 1) / (GHOST_TRAIL_LENGTH * 1.5)
                trail_layer = np.full(game_canvas.shape, TRAIL_COLOR_ON_BLUE_BG, dtype=np.uint8)
                trail_mask_bgr = cv2.cvtColor(gm_item, cv2.COLOR_GRAY2BGR)
                game_canvas = np.where(trail_mask_bgr == 255, 
                                       cv2.addWeighted(game_canvas, 1 - opacity, trail_layer, opacity, 0), 
                                       game_canvas)

    hue_shift_value = (hue_shift_value + 1) % 360

    display_debug = cv2.resize(debug_image_processed, (DISPLAY_W, DISPLAY_H))
    display_game = cv2.resize(game_canvas, (DISPLAY_W, DISPLAY_H))

    cv2.imshow(HELPER_WINDOW_NAME, display_debug)
    cv2.imshow(GAME_SCREEN_NAME, display_game)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

# Modalities

positive_emotion_list = [
    "Satisfaction",
    "Awe",
    "Contentment",
    "Concentration",
    "Excitement",
    "Interest",
    "Realization",
    "Desire",
    "Surprise (positive)",
    "Triumph",
]

negative_emotion_list = [
    "Distress",
    "Contemplation",
    "Confusion",
    "Doubt",
    "Surprise (negative)",
    "Anger",
    "Anxiety",
    "Awkwardness",
    "Contempt",
    "Disappointment",
    "Disgust",
    "Fear",
    "Boredom",
]

other_emotion_list = [
    "Admiration",
    "Adoration",
    "Aesthetic Appreciation",
    "Amusement",
    "Calmness",
    "Craving",
    "Determination",
    "Ecstasy",
    "Embarrassment",
    "Empathic Pain",
    "Entrancement",
    "Envy",
    "Guilt",
    "Horror",
    "Joy",
    "Love",
    "Nostalgia",
    "Pain",
    "Pride",
    "Relief",
    "Romance",
    "Sadness",
    "Shame",
    "Sympathy",
    "Tiredness"
]

emotion_list = positive_emotion_list + negative_emotion_list + other_emotion_list

emotion_list_basic = positive_emotion_list + negative_emotion_list

aus_list = [
    "AU1 Inner Brow Raise",
    "AU2 Outer Brow Raise",
    "AU4 Brow Lowerer",
    "AU5 Upper Lid Raise",
    "AU6 Cheek Raise",
    "AU7 Lids Tight",
    "AU9 Nose Wrinkle",
    "AU10 Upper Lip Raiser",
    "AU11 Nasolabial Furrow Deepener",
    "AU12 Lip Corner Puller",
    "AU14 Dimpler",
    "AU15 Lip Corner Depressor",
    "AU16 Lower Lip Depress",
    "AU17 Chin Raiser",
    "AU18 Lip Pucker",
    "AU19 Tongue Show",
    "AU20 Lip Stretch",
    "AU22 Lip Funneler",
    "AU23 Lip Tightener",
    "AU24 Lip Presser",
    "AU25 Lips Part",
    "AU26 Jaw Drop",
    "AU27 Mouth Stretch",
    "AU28 Lips Suck",
    "AU32 Bite",
    "AU34 Puff",
    "AU37 Lip Wipe",
    "AU38 Nostril Dilate",
    "AU43 Eye Closure",
    "AU53 Head Up",
    "AU54 Head Down",
]

other_expressions_list = [
    "Beaming",
    "Biting lip",
    "Cheering",
    "Cringe",
    "Face in hands",
    "Frown",
    "Gasp",
    "Glare",
    "Glaring",
    "Grimace",
    "Grin",
    "Jaw drop",
    "Laugh",
    "Licking lip",
    "Pout",
    "Scowl",
    "Smile",
    "Smirk",
    "Snarl",
    "Squint",
    "Sulking",
    "Tongue out",
    "Wide-eyed",
    "Wince",
    "Wrinkled nose",
]

gestures_list = [
    "Hand over Mouth",
    "Hand over Eyes",
    "Hand over Forehead",
    "Hand over Face",
    "Hand touching Face / Head",
    "Head Titling"
]

gaze_list = [
    "Gaze on Robot",
    "Gaze on Task",
    "Gaze on Misc"
]

body_pose_list = [
    "Head Tilting",
    "Crossed Arms",
    "Arms behind back"
]

gestures_list = other_expressions_list + gestures_list

hume = emotion_list + aus_list + other_expressions_list + gestures_list

all_modalities_list = hume + gaze_list + body_pose_list

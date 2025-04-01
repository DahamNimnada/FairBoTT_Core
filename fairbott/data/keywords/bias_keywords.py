#bias_keywords.py

BIAS_CATEGORIES = {
    "ability": [
        "disabled", "handicapped", "crippled", "differently-abled", "special needs", "wheelchair-bound",
        "physically challenged", "intellectually disabled", "neurodivergent", "autistic", "mentally ill",
        "visually impaired", "hearing impaired", "speech-impaired", "cognitively impaired"
    ],
    "race": [
        "black", "white", "asian", "indigenous", "mixed-race", "biracial", "people of color", "afro-caribbean",
        "afro-latino", "south asian", "east asian", "pacific islander", "native american", "aboriginal",
        "latin american", "middle eastern", "african-american", "hispanic"
    ],
    "gender": [
        "male", "female", "non-binary", "men", "women", "transgender", "cisgender", "genderfluid", "agender",
        "bigender", "two-spirit", "intersex", "genderqueer", "demiboy", "demigirl", "androgynous"
    ],
    "socioeconomic_class": [
        "poor", "wealthy", "middle class", "low-income", "working class", "upper class", "underprivileged",
        "privileged", "affluent", "blue-collar", "white-collar", "elite", "homeless", "disenfranchised"
    ],
    "body_type": [
        "fat", "thin", "obese", "overweight", "underweight", "slim", "muscular", "curvy", "petite", "plus-size",
        "skinny", "big-boned", "lanky", "stocky", "athletic build", "hourglass figure"
    ],
    "profession": [
        "doctor", "engineer", "teacher", "lawyer", "scientist", "nurse", "artist", "athlete", "entrepreneur",
        "politician", "actor", "musician", "writer", "journalist", "software developer", "blue-collar worker",
        "white-collar worker", "influencer", "military personnel", "civil servant"
    ],
    "cultural": [
        "tradition", "custom", "heritage", "ethnicity", "folklore", "native", "ancestry", "cultural background",
        "tribal", "nomadic", "colonial", "indigenous culture", "diaspora", "immigrant", "expatriate", "refugee"
    ],
    "disability": [
        "blind", "deaf", "mute", "autistic", "paralyzed", "mentally challenged", "speech-impaired", "neurological disorder",
        "bipolar", "schizophrenic", "depressive disorder", "epileptic", "cerebral palsy", "PTSD", "OCD", "ADHD"
    ],
    "race_ethnicity": [
        "hispanic", "caucasian", "african", "latino", "asian-american", "native american", "middle-eastern",
        "afro-european", "caribbean", "mediterranean", "indian", "arab", "eastern european", "nordic", "balkan"
    ],
    "characteristics": [
        "lazy", "aggressive", "weak", "arrogant", "emotional", "sensitive", "submissive", "dominant", "stubborn",
        "shy", "introverted", "extroverted", "hot-tempered", "manipulative", "passive", "unmotivated", "cold-hearted"
    ],
    "social": [
        "outcast", "elite", "privileged", "marginalized", "excluded", "disenfranchised", "mainstream", "minority",
        "upper-class", "lower-class", "social climber", "popular", "ostracized", "rebel", "counterculture"
    ],
    "victim": [
        "harassed", "oppressed", "abused", "exploited", "bullied", "persecuted", "discriminated against", "silenced",
        "mistreated", "assaulted", "victimized", "marginalized", "targeted", "wrongfully accused"
    ],
    "religion": [
        "christian", "muslim", "hindu", "buddhist", "jewish", "atheist", "agnostic", "sikh", "pagan",
        "protestant", "catholic", "orthodox christian", "evangelical", "shinto", "taoist", "zoroastrian"
    ],
    "political_ideologies": [
        "liberal", "conservative", "socialist", "communist", "capitalist", "libertarian", "progressive", "moderate",
        "anarchist", "fascist", "populist", "authoritarian", "nationalist", "leftist", "right-wing", "centrist"
    ],
    "nationality": [
        "american", "british", "chinese", "french", "german", "indian", "russian", "japanese", "korean", "brazilian",
        "italian", "mexican", "canadian", "spanish", "nigerian", "egyptian", "iranian", "turkish", "indonesian",
        "argentinian", "south african", "pakistani", "australian", "thai", "vietnamese"
    ]
}

BIAS_TYPES = [
    "ability", "body_type", "characteristics", "cultural", "disabled", "gender",
    "nationality", "neutral", "political_ideologies", "profession", "race",
    "race_ethnicity", "religion", "social", "socioeconomic_class", "unknown",
    "victim", "bias_general", "other"
]
"""
Multilingual Voice Assistant for KisanQ (Telugu & Pleasant English Female/Girl Voice).
Provides spoken agricultural guidance for both Mango and Hibiscus crops.
"""

import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from gtts import gTTS

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# Comprehensive Agricultural Advisory Database for Multi-Crop (Mango & Hibiscus)
ADVISORY_DATABASE = {
    # --- Mango Diagnoses ---
    "Healthy": {
        "en": {
            "title": "Healthy Foliage (ఆరోగ్యకరమైన ఆకు)",
            "speech": (
                "Hello farmer! Good news! Your plant leaf looks wonderfully vibrant and healthy. "
                "The chlorophyll levels are strong, and there are no signs of infection. "
                "Keep up your regular watering schedule and inspect your crops weekly!"
            ),
            "advice": [
                "Maintain balanced soil nutrition and regular drip irrigation.",
                "Ensure proper tree canopy pruning for maximum sunlight penetration.",
                "Continue weekly preventive leaf inspections."
            ]
        },
        "te": {
            "title": "ఆరోగ్యకరమైన ఆకు",
            "speech": (
                "నమస్కారం రైతు సోదరులారా. మీ మొక్క ఆకు పూర్తిగా ఆరోగ్యంగా ఉంది. ఎలాంటి వ్యాధి లక్షణాలు కనిపించడం లేదు. "
                "క్రమం తప్పకుండా డ్రిప్ నీరు, సేంద్రీయ ఎరువులు అందించండి మరియు ప్రతి వారం గమనించండి."
            ),
            "advice": [
                "సమతుల్య సేంద్రీయ ఎరువులు మరియు డ్రిప్ నీటి పారుదల కొనసాగించండి.",
                "చెట్టుకు తగినంత గాలి, వెలుతురు సోకేలా కొమ్మలను సరిగ్గా కత్తిరించండి.",
                "ప్రతి వారం కొత్త చిగుర్లను జాగ్రత్తగా పరిశీలించండి."
            ]
        }
    },
    "Anthracnose": {
        "en": {
            "title": "Anthracnose Fungal Blight (ఆంత్రాక్నోస్ తెగులు)",
            "speech": (
                "Hello farmer. I detected dark brown necrotic spots on the leaf surface, indicating Anthracnose fungal blight. "
                "Please avoid overhead watering to prevent spore spreading, prune any infected twigs, "
                "and consult your local agriculture officer for safe organic copper treatment."
            ),
            "advice": [
                "Prune and destroy infected fallen leaves and dead twigs to prevent fungal spore spread.",
                "Avoid overhead irrigation; wet leaves accelerate fungal sporulation.",
                "Consult the local Agricultural Officer (AEO/KVK) for recommended copper hydroxide spray."
            ]
        },
        "te": {
            "title": "ఆంత్రాక్నోస్ (తెగులు)",
            "speech": (
                "రైతు సోదరులారా గమనించండి. మీ మామిడి ఆకులో ఆంత్రాక్నోస్ తెగులు లక్షణాలు కనిపిస్తున్నాయి. "
                "నల్లటి మచ్చలు ఏర్పడ్డాయి. పైనుంచి నీరు చల్లవద్దు. "
                "దయచేసి తక్షణమే దగ్గరలోని కృషి విజ్ఞాన కేంద్రం లేదా వ్యవసాయ నిపుణుడిని సంప్రదించండి."
            ),
            "advice": [
                "తెగులు సోకిన ఆకులను, కొమ్మలను కత్తిరించి కాల్చివేయండి.",
                "పైనుంచి నీరు చల్లవద్దు; ఆకులు తడిగా ఉంటే శిలీంధ్రం వేగంగా వ్యాపిస్తుంది.",
                "సిఫార్సు చేసిన కాపర్ లేదా వేప నూనె పిచికారీ కోసం వ్యవసాయ అధికారిని సంప్రదించండి."
            ]
        }
    },
    "Bacterial Canker": {
        "en": {
            "title": "Bacterial Canker Lesion (బాక్టీరియల్ క్యాంకర్)",
            "speech": (
                "Hello farmer. The leaf shows water-soaked lesions surrounded by yellow halos, which looks like Bacterial Canker. "
                "Always disinfect your pruning shears with alcohol, and check in with your district extension expert."
            ),
            "advice": [
                "Disinfect all pruning shears in 10% bleach or rubbing alcohol.",
                "Do not prune during rain or high wind to avoid spreading bacterial ooze.",
                "Consult your local agricultural university or KVK extension agent for bactericidal protocols."
            ]
        },
        "te": {
            "title": "బాక్టీరియల్ క్యాంకర్",
            "speech": (
                "రైతు సోదరులారా జాగ్రత్త. మీ మామిడి ఆకులో బాక్టీరియల్ క్యాంకర్ వ్యాధి లక్షణాలు కనిపిస్తున్నాయి. "
                "ఆకులపై పసుపు రంగు వలయాలతో మచ్చలు ఉన్నాయి. "
                "కత్తిరించే పనిముట్లను శానిటైజ్ చేయండి మరియు స్థానిక వ్యవసాయ నిపుణుడిని సంప్రదించండి."
            ),
            "advice": [
                "కత్తిరించే కత్తెరలను ఆల్కహాల్ లేదా డెట్టాల్‌తో శుభ్రం చేయండి.",
                "వర్షం లేదా గాలులు ఉన్నప్పుడు కొమ్మలను కత్తిరించవద్దు.",
                "సరైన రక్షణ చర్యల కోసం సమీప వ్యవసాయ విస్తరణ అధికారిని కలవండి."
            ]
        }
    },

    # --- Hibiscus Diagnoses ---
    "Leaf Spot / Blight": {
        "en": {
            "title": "Hibiscus Leaf Spot / Blight (మందార ఆకు మచ్చల తెగులు)",
            "speech": (
                "Hello farmer! I inspected your hibiscus foliage and noticed leaf spot and fungal blight symptoms. "
                "Remove and bag spotted leaves promptly, water only at the soil base, "
                "and apply an organic neem oil spray in the cool evening."
            ),
            "advice": [
                "Gently strip heavily spotted bottom leaves and dispose of them away from the plant.",
                "Water at the base of the soil in the morning; do not wet the hibiscus blooms or foliage.",
                "Apply cold-pressed organic neem oil spray (5ml per liter of water) during early morning or sunset."
            ]
        },
        "te": {
            "title": "మందార ఆకు మచ్చల తెగులు",
            "speech": (
                "నమస్కారం రైతు సోదరులారా. మీ మందార ఆకులో మచ్చల తెగులు లక్షణాలు ఉన్నాయి. "
                "మచ్చలు పడిన ఆకులను తొలగించండి. కేవలం మొదట్లోనే నీరు పోయండి. "
                "సాయంత్రం వేళల్లో వేప నూనె లేదా సేంద్రీయ మందులను పిచికారీ చేయండి."
            ),
            "advice": [
                "మచ్చలు ఎక్కువగా ఉన్న ఆకులను వెంటనే తెంపి దూరంగా పారవేయండి.",
                "మొక్క మొదట్లోనే నీరు పోయండి; పూలు మరియు ఆకులపై నీరు చల్లవద్దు.",
                "సేంద్రీయ వేప నూనెను లీటరు నీటికి 5 మిల్లీలీటర్లు కలిపి సాయంత్రం పిచికారీ చేయండి."
            ]
        }
    }
}


class VoiceAssistantEngine:
    """Manages bilingual audio generation with high-quality female English voice."""

    def __init__(self, audio_dir: Optional[Path] = None):
        self.audio_dir = Path(audio_dir) if audio_dir else config.AUDIO_DIR
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def get_advisory(self, disease_class: str, lang: str = "te") -> Dict[str, Any]:
        """Fetch structured advisory text in target language."""
        disease_info = ADVISORY_DATABASE.get(disease_class, ADVISORY_DATABASE["Healthy"])
        lang_code = lang if lang in ["te", "en"] else "te"
        return disease_info.get(lang_code, disease_info["en"])

    def generate_audio_file(self, text: str, lang: str = "en", filename: Optional[str] = None) -> Path:
        """
        Synthesize speech using gTTS with feminine accent/tone parameters (tld='com' or 'co.uk').
        """
        if filename is None:
            text_hash = hashlib.md5(f"{lang}_{text}".encode("utf-8")).hexdigest()[:10]
            filename = f"speech_{lang}_{text_hash}.mp3"

        out_path = self.audio_dir / filename
        if not out_path.exists():
            try:
                # English voice with clean British/American female cadence
                tld = "co.uk" if lang == "en" else "com"
                tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
                tts.save(str(out_path))
                print(f"[VoiceAssistant] Generated audio file: {out_path.name}")
            except Exception as e:
                print(f"[VoiceAssistant] TTS warning: {e}")
        return out_path

    def prebuild_standard_audio_cache(self):
        """Pre-generates and caches all standard Mango and Hibiscus advisories."""
        print("[VoiceAssistant] Caching bilingual audio files for Mango and Hibiscus...")
        for disease, lang_map in ADVISORY_DATABASE.items():
            for lang, data in lang_map.items():
                safe_name = disease.lower().replace(" ", "_").replace("/", "_")
                filename = f"{safe_name}_{lang}.mp3"
                self.generate_audio_file(text=data["speech"], lang=lang, filename=filename)


if __name__ == "__main__":
    engine = VoiceAssistantEngine()
    engine.prebuild_standard_audio_cache()
    adv = engine.get_advisory("Leaf Spot / Blight", "en")
    print("English Girl Voice Advisory Sample:")
    print(adv["speech"])

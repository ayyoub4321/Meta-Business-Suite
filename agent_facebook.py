import requests

import os,json
import re
import string
import urllib.request
from pathlib import Path

import nltk
import matplotlib
matplotlib.use("Agg")  # Fonctionne sans interface graphique

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import time
import requests
from pathlib import Path
from PIL import Image
import cloudinary
import cloudinary.uploader
from langchain_groq import ChatGroq
from groq import Groq
import feedparser
import torch
from diffusers import AutoPipelineForText2Image
from datetime import datetime, timezone


TRUSTED_WEB_SOURCES = {
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "InvestingLive Macro": "https://investinglive.com/rss/news",
    "ActionForex Fundamental": (
        "https://www.actionforex.com/category/fundamental-analysis/feed/"
    ),
}
PAGE_ID=os.getenv("PAGE_ID", globals().get("PAGE_ID", ""))
INSTAGRAM_BUSINESS_ID=os.getenv("INSTAGRAM_BUSINESS_ID", globals().get("INSTAGRAM_BUSINESS_ID", ""))
API_VERSION = "v21.0"

# URL de base de l'API Graph de Meta
FB_BASE_URL = f"https://graph.facebook.com/{API_VERSION}/{PAGE_ID}"

IG_BASE_URL = f"https://graph.facebook.com/{API_VERSION}/{INSTAGRAM_BUSINESS_ID}"

# Utilisée par les fonctions de publication photo et vidéo
BASE_URL = FB_BASE_URL






# =====================================================
# CHEMINS PORTABLES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
OUTPUT_DIR = BASE_DIR / "output"
NLTK_DIR = BASE_DIR / ".data" / "nltk"

for dossier in (FONTS_DIR, OUTPUT_DIR, NLTK_DIR):
    dossier.mkdir(parents=True, exist_ok=True)

if str(NLTK_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DIR))





class TextPost:

    def __init__(self):
        model_groq_name="qwen/qwen3.8-27b"
        # رفعنا الـ temperature باش الـ LLM يكون مبدع وتتنوع الأفكار وما يبقاش مبلواني
        self.llm = ChatGroq(
            model_name=model_groq_name,
            temperature=0.8,
            groq_api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", globals().get("TOKEN_AI_DESCRYB_IMG", "")) 
        )
        
        model_id = "stabilityai/sdxl-turbo"

        # 1. Configuration du matériel
        if torch.cuda.is_available():
            device, dtype, variant = "cuda", torch.float16, "fp16"
        elif torch.backends.mps.is_available():
            device, dtype, variant = "mps", torch.float16, "fp16"
        else:
            device, dtype, variant = "cpu", torch.float32, None

        # 2. Chargement du modèle
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            variant=variant
        ).to(device)

    def donne_moi_prompt_images(self, news):
        system_prompt = """
        You are an elite, world-class Hollywood 3D Art Director and Master Visual Prompt Engineer for premium financial journalism. 
        Your absolute superpower is transforming dry financial news articles into hyper-realistic, striking, dynamic 3D visual metaphors that tell a compelling story at a glance.
        """
        
        user_prompt = f"""
        Analyze the following financial news article with extreme care. Extract the unique economic event, trend, or decision (e.g., interest rate pause, inflation spikes, central bank policy shifts, market crashes, or currency surges). 
        Then, create a completely unique, highly relevant, and breathtaking 3D visual scene that represents THIS specific news and nothing else. Never use repetitive generic clichés (like old anchors or basic shields). Make it dynamic, original, and deeply connected to the text.

        NEWS ARTICLE:
        {news}

        STRICT GENERATION RULES:
        1. UNIQUE VISUAL METAPHOR: Invent custom, engaging 3D elements (e.g., floating kinetic energy pendulums, futuristic digital data crystals, glowing macroeconomic balancing mechanisms, or abstract financial wave architectures) matching the exact sentiment of the news.
        2. CINEMATIC AESTHETIC & QUALITY:
        - Style: Unreal Engine 5 hyper-realistic 3D render, Octane render, crystal-clear macro and isometric details.
        - Background: Deep moody cinematic background (dark navy, obsidian slate #05070B) with subtle ambient floating particles.
        - Lighting: Dramatic volumetric neon rim lighting (electric blue, emerald green, or fiery amber matching the specific news vibe), high-contrast ray-traced studio reflections.
        3. ABSOLUTE CONSTRAINTS:
        - NO text, letters, numbers, symbols, words, watermarks, signatures, human faces, or anatomy anywhere.

        OUTPUT FORMAT (STRICT):
        Output ONLY the two blocks below without introduction or conclusion:

        POSITIVE PROMPT:
        [A hyper-detailed, cinematic 3D paragraph in English describing precise lighting, premium materials like brushed titanium and glowing frosted glass, dramatic composition, and dynamic motion tailored explicitly to this news]

        NEGATIVE PROMPT:
        text, letters, words, numbers, watermark, signature, human, face, blurry, low resolution, noise, distortion, 2d illustration, flat, cartoon, boring, generic anchor
        """

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return response
        except Exception as e:
            print(f"Erreur LLM prompt generation: {e}")
            class Fallback:
                content = "POSITIVE PROMPT:\nFuturistic 3D abstract financial data architecture, glowing neon lights, cinematic 8k render\nNEGATIVE PROMPT:\ntext, letters, words, numbers, watermark"
            return Fallback()

    def generer_image(self, news_original=None):
        print(" Génération de l'image dynamique mkatba 3la lkhabar...")
        obj_describe = self.donne_moi_prompt_images(news_original)
        text_describe = obj_describe.content
        
        try:
            positive_prompt = text_describe.split("NEGATIVE PROMPT:")[1].strip()
            negative_prompt = text_describe.split("NEGATIVE PROMPT:")[0].split("POSITIVE PROMPT:")[1].strip()
        except Exception:
            positive_prompt = "Futuristic 3D abstract financial data architecture, glowing neon lights, hyper-detailed render"
            negative_prompt = "text, letters, words, numbers, watermark, signature, human, face, blurry"

        image = self.pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,  # الجودة و الدقة
            guidance_scale=1.2      # باش الموديل يلتزم بدقة بالبرومبت وما يخرجش عليه
        ).images[0]

        output = OUTPUT_DIR / "img.png"
        image.save(output)
        print(f"Terminé ! L'image dynamique est sauvegardée sous {output}")
        return output
        

    
    def publier_texte_partout(self, news_original,news_translited):
        

        resultats = {
            "image": None,
            "facebook": False,
            "instagram": False
        }

        try:
    
            print("🎨 Génération de l'image...")

            imgs_path = self.generer_image(news_original=news_original)
            chemin = Path(imgs_path)

            if not chemin.is_file():
                raise FileNotFoundError(
                    f"Image générée introuvable : {chemin.resolve()}"
                )

            resultats["image"] = str(chemin.resolve())
            print(f"✅ Image générée : {chemin.resolve()}")

            # ==========================================
            # 2. VÉRIFIER LA CONFIGURATION
            # ==========================================
            page_id = os.getenv("PAGE_ID", globals().get("PAGE_ID", ""))
            token = os.getenv("PAGE_ACCESS_TOKEN", globals().get("PAGE_ACCESS_TOKEN", ""))
            ig_id = os.getenv(
                "INSTAGRAM_BUSINESS_ID",
                globals().get("INSTAGRAM_BUSINESS_ID", "")
            )

            cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", globals().get("CLOUDINARY_CLOUD_NAME", "")) 
            cloud_key =  os.getenv("CLOUDINARY_API_KEY", globals().get("CLOUDINARY_API_KEY", ""))
            cloud_secret = os.getenv("CLOUDINARY_API_SECRET", globals().get("CLOUDINARY_API_SECRET", ""))

            # Compatibilité avec les variables existantes du projet
            base_url = globals().get("BASE_URL", "")
            if not page_id and base_url:
                page_id = base_url.rstrip("/").split("/")[-1]

            if not token:
                raise ValueError(
                    "PAGE_ACCESS_TOKEN est absent. Configure le jeton de ta Page."
                )

            graph_url = "https://graph.facebook.com/v26.0"

            # ==========================================
            # 3. PUBLIER SUR FACEBOOK
            # ==========================================
            print("\n📢 Publication Facebook...")

            if not page_id:
                print("❌ FACEBOOK_PAGE_ID ou PAGE_ID manquant.")
            else:
                endpoint_fb = f"{graph_url}/{page_id}/photos"

                mime = (
                    "image/jpeg"
                    if chemin.suffix.lower() in (".jpg", ".jpeg")
                    else "image/png"
                )

                try:
                    with chemin.open("rb") as fichier:
                        response = requests.post(
                            endpoint_fb,
                            data={
                                "message": news_translited,
                                "access_token": token
                            },
                            files={
                                "source": (chemin.name, fichier, mime)
                            },
                            timeout=120
                        )

                    try:
                        data_fb = response.json()
                    except ValueError:
                        data_fb = {"message": response.text[:1000]}

                    if response.ok and "error" not in data_fb:
                        resultats["facebook"] = True
                        print("✅ Facebook : publication réussie.")
                        print(
                            "ID :",
                            data_fb.get("post_id", data_fb.get("id"))
                        )
                    else:
                        print("❌ Facebook :", data_fb)

                except requests.RequestException as erreur:
                    print("❌ Erreur réseau Facebook :", erreur)

            # ==========================================
            # 4. HÉBERGER L'IMAGE POUR INSTAGRAM
            # ==========================================
            print("\n☁️ Préparation de l'image pour Instagram...")

            if not all([cloud_name, cloud_key, cloud_secret]):
                print(
                    "❌ Instagram : identifiants Cloudinary manquants. "
                    "Configure CLOUDINARY_CLOUD_NAME, "
                    "CLOUDINARY_API_KEY et CLOUDINARY_API_SECRET."
                )
            elif not ig_id:
                print(
                    "❌ Instagram : INSTAGRAM_BUSINESS_ID manquant."
                )
            else:
                try:
                    cloudinary.config(
                        cloud_name=cloud_name,
                        api_key=cloud_key,
                        api_secret=cloud_secret,
                        secure=True
                    )

                    # Instagram accepte ici une image JPEG.
                    jpeg_path = chemin.with_name(
                        f"{chemin.stem}_instagram.jpg"
                    )

                    with Image.open(chemin) as image:
                        image.convert("RGB").save(
                            jpeg_path,
                            format="JPEG",
                            quality=92,
                            optimize=True
                        )

                    upload = cloudinary.uploader.upload(
                        str(jpeg_path),
                        resource_type="image",
                        folder="actualites_financieres"
                    )

                    image_url = upload.get("secure_url")

                    if not image_url or not image_url.startswith("https://"):
                        raise RuntimeError(
                            "Cloudinary n'a pas retourné d'URL HTTPS publique."
                        )

                    print("✅ Image hébergée pour Instagram.")

                    # ==========================================
                    # 5. CRÉER LE CONTENEUR INSTAGRAM
                    # ==========================================
                    print("📢 Création du post Instagram...")

                    creation_response = requests.post(
                        f"{graph_url}/{ig_id}/media",
                        data={
                            "image_url": image_url,
                            "caption": news_translited,
                            "access_token": token
                        },
                        timeout=90
                    )

                    try:
                        creation_data = creation_response.json()
                    except ValueError:
                        creation_data = {
                            "message": creation_response.text[:1000]
                        }

                    if (
                        not creation_response.ok
                        or "error" in creation_data
                        or not creation_data.get("id")
                    ):
                        print("❌ Instagram - création :", creation_data)
                    else:
                        creation_id = creation_data["id"]

                        # Attendre que Meta traite l'image.
                        statut = None

                        for tentative in range(12):
                            statut_response = requests.get(
                                f"{graph_url}/{creation_id}",
                                params={
                                    "fields": "status_code",
                                    "access_token": token
                                },
                                timeout=30
                            )

                            try:
                                statut_data = statut_response.json()
                            except ValueError:
                                statut_data = {}

                            if not statut_response.ok or "error" in statut_data:
                                print(
                                    "❌ Instagram - vérification :",
                                    statut_data
                                )
                                break

                            statut = statut_data.get("status_code")

                            if statut == "FINISHED":
                                break

                            if statut == "ERROR":
                                print(
                                    "❌ Instagram n'a pas pu traiter l'image."
                                )
                                break

                            time.sleep(5)

                        if statut == "FINISHED":
                            # ==========================================
                            # 6. PUBLIER LE POST INSTAGRAM
                            # ==========================================
                            publication_response = requests.post(
                                f"{graph_url}/{ig_id}/media_publish",
                                data={
                                    "creation_id": creation_id,
                                    "access_token": token
                                },
                                timeout=90
                            )

                            try:
                                publication_data = publication_response.json()
                            except ValueError:
                                publication_data = {
                                    "message": publication_response.text[:1000]
                                }

                            if (
                                publication_response.ok
                                and "error" not in publication_data
                                and publication_data.get("id")
                            ):
                                resultats["instagram"] = True
                                print("✅ Instagram : publication réussie.")
                                print("ID :", publication_data["id"])
                            else:
                                print(
                                    "❌ Instagram - publication :",
                                    publication_data
                                )
                        elif statut != "ERROR":
                            print(
                                "❌ Instagram : délai de traitement dépassé."
                            )

                except Exception as erreur:
                    print("❌ Erreur Instagram :", erreur)

        except Exception as erreur:
            print("❌ Erreur de génération ou de préparation :", erreur)

        # ==========================================
        # 7. RÉSUMÉ
        # ==========================================
        print("\n========== RÉSULTAT FINAL ==========")
        print("Image :", resultats["image"])
        print("Facebook :", "✅ Publié" if resultats["facebook"] else "❌ Échec")
        print("Instagram :", "✅ Publié" if resultats["instagram"] else "❌ Échec")

        return resultats

class Extract_post:
    def __init__(self):
            try:
                GROQ_API_KEY = os.getenv("TOKEN_AI_DESCRYB_IMG", globals().get("TOKEN_AI_DESCRYB_IMG", ""))
                self.model_groq_name="qwen/qwen3.8-27b"
                self.client = Groq(api_key=GROQ_API_KEY)
                self.TRUSTED_WEB_SOURCES = {
                "FXStreet": "https://www.fxstreet.com/rss/news",
                "InvestingLive Macro": "https://investinglive.com/rss/news",
                "ActionForex Fundamental": (
                    "https://www.actionforex.com/category/fundamental-analysis/feed/"
                ),
                    }
            except Exception as e:
                print("errur lorsque creation de objet Extract post ",{e})
    def clean_html(self,text):
        """Supprime les balises HTML sans reformuler le texte."""
        if not text:
            return ""

        soup = BeautifulSoup(text, "html.parser")

        # Supprimer les éléments qui ne sont pas du contenu éditorial.
        for element in soup(["script", "style"]):
            element.decompose()

        # Extraire le texte et décoder les entités HTML.
        return soup.get_text(separator=" ", strip=True)

    def get_publication_date(self,entry):
        """Utilise la date de publication fournie par le flux RSS."""

        published = entry.get("published_parsed")
        updated = entry.get("updated_parsed")

        # Priorité à la date de publication, jamais à la date d'extraction.
        date_struct = published or updated

        if not date_struct:
            return None, None

        date_utc = datetime(
            *date_struct[:6],
            tzinfo=timezone.utc
        )

        # Date affichée à la minute, en UTC.
        date_text = date_utc.strftime("%Y-%m-%d %H:%M")

        return date_text, date_utc

    

    def fetch_all_news(self):
        articles = []
        seen_links = set()

        for source_name, feed_url in self.TRUSTED_WEB_SOURCES.items():
            try:
                feed = feedparser.parse(feed_url)

                if getattr(feed, "bozo", False):
                    # print(f"Attention : flux possiblement invalide : {source_name}")
                    pass

                if getattr(feed, "status", 200) >= 400:
                    # print(f"Erreur HTTP : {source_name}")
                    continue

                for entry in feed.entries:
                    link = entry.get("link", "").strip()
                    title = entry.get("title", "").strip()

                    if not title or not link:
                        continue

                    if link in seen_links:
                        continue

                    # Le contenu original du flux, sans reformulation.
                    raw_text = (
                        entry.get("content", [{}])[0].get("value", "")
                        if entry.get("content")
                        else entry.get("summary", "")
                    )

                    clean_text = self.clean_html(raw_text)

                    publication_text, publication_datetime = (
                        self.get_publication_date(entry)
                    )

                    # Ne pas inventer une date si elle est absente.
                    if publication_text is None:
                        # print(f"Date de publication absente : {title}")
                        continue

                    seen_links.add(link)

                    articles.append({
                        "title": self.clean_html(title),
                        "date": publication_text,
                        "continue": clean_text,
                        "_datetime": publication_datetime,
                    })
            except Exception as error:
                print(f"Erreur pour {source_name}: {error}")
                pass
        # Les articles les plus récents en premier.
        articles.sort(
            key=lambda item: item["_datetime"],
            reverse=True
        )
        # Structure JSON demandée : source1, source2, source3...
        result = {}
        for index, article in enumerate(articles, start=1):
            article.pop("_datetime")
            result[f"{source_name}_{article['date']}"] = article
        return result

    
    def tradication_of_text(self, text):
        prompt = """
            You are a professional Arabic translator specializing in financial news.

            Translate the source text into natural, professional Modern Standard Arabic.
            Preserve the financial meaning, figures, dates, names, and facts.
            Do not invent information.

            Return ONLY a valid JSON object using this exact structure:
             {'caption':text_arabe. + source + date }'

            The response must be valid JSON.
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model_groq_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"SOURCE TEXT TO TRANSLATE:\n{text}"
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            raw_output = response.choices[0].message.content
            result = json.loads(raw_output)

            if not isinstance(result, dict) or not isinstance(
                result.get("caption"), str
            ):
                raise ValueError("Le JSON doit contenir un champ caption textuel.")

            return result

        except Exception as e:
            print(f"Erreur lors de l'appel Groq : {e}")
            return None



        

import os
import json
import time


def main():
    fichier_json = "historique_posts.json"
    extract_post = Extract_post()
    text_post = TextPost()

    historique = {}

    if os.path.exists(fichier_json):
        with open(fichier_json, "r", encoding="utf-8") as f:
            try:
                historique = json.load(f)
            except json.JSONDecodeError:
                historique = {}

    fetch_all_news = extract_post.fetch_all_news()
    changement_detecte = False

    for source, news in fetch_all_news.items():
        vrai_source = source.split("_")[0]
        title = news["title"]
        content = news["continue"]
        date = news["date"]

        post_id = f"{vrai_source}_{title}_{date}"

        if post_id in historique:
            print(f"Déjà traité : {title}")
            continue

        text_original = (
            f"{title}\n{content}\n\n"
            f"source: {vrai_source}, date: {date}"
        )

        text_transleted = extract_post.tradication_of_text(text_original)

        if isinstance(text_transleted, dict):
            text_transleted_net = " ".join(
                str(value) for value in text_transleted.values()
            )
        else:
            text_transleted_net = str(text_transleted)

        text_post.publier_texte_partout(
            text_original, text_transleted_net
        )

        historique[post_id] = {
            "source": vrai_source,
            "titre": title,
            "date": date
        }
        changement_detecte = True

    if changement_detecte:
        with open(fichier_json, "w", encoding="utf-8") as f:
            json.dump(historique, f, ensure_ascii=False, indent=4)

    print("Fin de cette exécution.")


if __name__ == "__main__":
    main()

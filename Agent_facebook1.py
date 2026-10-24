import os
import json
import re
import string
import urllib.request
from pathlib import Path

import nltk
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import time
import requests
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
    "ActionForex Fundamental": "https://www.actionforex.com/category/fundamental-analysis/feed/",
}

PAGE_ID = os.getenv("PAGE_ID", "")
INSTAGRAM_BUSINESS_ID = os.getenv("INSTAGRAM_BUSINESS_ID", "")
API_VERSION = "v21.0"

FB_BASE_URL = f"https://graph.facebook.com/{API_VERSION}/{PAGE_ID}"
IG_BASE_URL = f"https://graph.facebook.com/{API_VERSION}/{INSTAGRAM_BUSINESS_ID}"
BASE_URL = FB_BASE_URL

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
        model_groq_name = "qwen/qwen3.8-27b"
        self.llm = ChatGroq(
            model_name=model_groq_name,
            temperature=0.8,
            groq_api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", "")
        )
        
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        if torch.cuda.is_available():
            device, dtype, variant = "cuda", torch.float16, "fp16"
        elif torch.backends.mps.is_available():
            device, dtype, variant = "mps", torch.float16, "fp16"
        else:
            device, dtype, variant = "cpu", torch.float32, None

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            variant=variant
        ).to(device)

    def donne_moi_prompt_images(self, news):
        system_prompt = """
        You are an uncompromising, world-class Financial Photojournalist and Editorial Art Director.
        Your sole task is to generate realistic, accurate visual prompts for real-world financial news photos.
        You strictly forbid abstract designs, 3D wireframes, glowing neon elements, and generic futuristic shapes.
        """
        
        user_prompt = f"""
        ANALYZE THIS FINANCIAL NEWS CAREFULLY:
        "{news}"

        STEP 1: IDENTIFY THE REAL-WORLD ASSETS & SUBJECTS
        Identify the concrete entities mentioned in the news:
        - Currencies (e.g., stacks of US Dollar bills, Euro banknotes, Gold bullion bars)
        - Commodities (e.g., crude oil barrels, energy grids, agricultural goods)
        - Institutions & Buildings (e.g., Federal Reserve headquarters, Wall Street stock exchange floor, Central Bank vaults)
        - Market Trends (e.g., stock market trading screens showing green or red chart candles)

        STEP 2: CREATE A PHOTOREALISTIC SCENE
        Describe a real-world, high-end editorial photo suitable for Bloomberg or Reuters news headers.

        STRICT RULES & CONSTRAINTS:
        1. REALISM ONLY: Must look like a real photo shot with a 85mm lens, f/1.8 aperture, natural cinematic lighting.
        2. NO ABSTRACT ELEMENTS: Zero 3D floating shapes, zero wireframes, zero glowing neon lines, zero abstract energy pendulums.
        3. NO TEXT/FACES: NO text, words, numbers, letters, logos, watermarks, human faces, or human figures.

        STRICT OUTPUT FORMAT (Output ONLY these two lines):
        POSITIVE PROMPT:
        [Photorealistic description of concrete financial items matching the news context, captured with professional editorial photography, sharp focus, dramatic studio lighting]

        NEGATIVE PROMPT:
        abstract, 3d render, floating shapes, wireframe, neon lights, generic geometric shapes, text, letters, words, numbers, watermark, signature, human, face, blurry, cartoon, illustration
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
                content = "POSITIVE PROMPT:\nA realistic financial photo of high-resolution stock market charts on trading monitors, stacks of cash bills, cinematic lighting, editorial style\nNEGATIVE PROMPT:\nabstract, 3d render, neon, text, letters, human, face, watermark"
            return Fallback()
    def generer_image(self, news_original=None):
        print("Génération de l'image dynamique...")
        obj_describe = self.donne_moi_prompt_images(news_original)
        text_describe = obj_describe.content
        
        try:
            positive_prompt = text_describe.split("NEGATIVE PROMPT:")[1].strip()
            negative_prompt = text_describe.split("NEGATIVE PROMPT:")[0].split("POSITIVE PROMPT:")[1].strip()
        except Exception:
            positive_prompt = "Futuristic 3D abstract financial data architecture, glowing neon lights"
            negative_prompt = "text, letters, words, numbers, watermark, signature, human, face, blurry"

        image = self.pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,
            guidance_scale=1.2
        ).images[0]

        output = OUTPUT_DIR / "img.png"
        image.save(output)
        print(f"Terminé ! L'image dynamique est sauvegardée sous {output}")
        return output

    def publier_texte_partout(self, news_original, news_translited):
        resultats = {
            "image": None,
            "facebook": False,
            "instagram": False
        }

        try:
            print("Génération de l'image...")
            imgs_path = self.generer_image(news_original=news_original)
            chemin = Path(imgs_path)

            if not chemin.is_file():
                raise FileNotFoundError(f"Image générée introuvable : {chemin.resolve()}")

            resultats["image"] = str(chemin.resolve())

            page_id = os.getenv("PAGE_ID", "")
            token = os.getenv("PAGE_ACCESS_TOKEN", "")
            ig_id = os.getenv("INSTAGRAM_BUSINESS_ID", "")

            cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "")
            cloud_key = os.getenv("CLOUDINARY_API_KEY", "")
            cloud_secret = os.getenv("CLOUDINARY_API_SECRET", "")

            if not token:
                raise ValueError("PAGE_ACCESS_TOKEN est absent.")

            graph_url = "https://graph.facebook.com/v26.0"

            # Facebook
            if page_id:
                endpoint_fb = f"{graph_url}/{page_id}/photos"
                mime = "image/jpeg" if chemin.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                try:
                    with chemin.open("rb") as fichier:
                        response = requests.post(
                            endpoint_fb,
                            data={"message": news_translited, "access_token": token},
                            files={"source": (chemin.name, fichier, mime)},
                            timeout=120
                        )
                    data_fb = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"message": response.text[:1000]}
                    if response.ok and "error" not in data_fb:
                        resultats["facebook"] = True
                        print("Facebook : publication réussie.")
                except Exception as erreur:
                    print("Erreur Facebook :", erreur)

            # Instagram
            if all([cloud_name, cloud_key, cloud_secret]) and ig_id:
                try:
                    cloudinary.config(
                        cloud_name=cloud_name,
                        api_key=cloud_key,
                        api_secret=cloud_secret,
                        secure=True
                    )
                    jpeg_path = chemin.with_name(f"{chemin.stem}_instagram.jpg")
                    with Image.open(chemin) as image:
                        image.convert("RGB").save(jpeg_path, format="JPEG", quality=92, optimize=True)

                    upload = cloudinary.uploader.upload(str(jpeg_path), resource_type="image", folder="actualites_financieres")
                    image_url = upload.get("secure_url")

                    if image_url and image_url.startswith("https://"):
                        creation_response = requests.post(
                            f"{graph_url}/{ig_id}/media",
                            data={"image_url": image_url, "caption": news_translited, "access_token": token},
                            timeout=90
                        )
                        creation_data = creation_response.json()
                        if creation_response.ok and "error" not in creation_data and creation_data.get("id"):
                            creation_id = creation_data["id"]
                            statut = None
                            for _ in range(12):
                                statut_response = requests.get(
                                    f"{graph_url}/{creation_id}",
                                    params={"fields": "status_code", "access_token": token},
                                    timeout=30
                                )
                                statut_data = statut_response.json()
                                statut = statut_data.get("status_code")
                                if statut == "FINISHED" or statut == "ERROR":
                                    break
                                time.sleep(5)

                            if statut == "FINISHED":
                                pub_response = requests.post(
                                    f"{graph_url}/{ig_id}/media_publish",
                                    data={"creation_id": creation_id, "access_token": token},
                                    timeout=90
                                )
                                pub_data = pub_response.json()
                                if pub_response.ok and "error" not in pub_data:
                                    resultats["instagram"] = True
                                    print("Instagram : publication réussie.")
                except Exception as erreur:
                    print("Erreur Instagram :", erreur)

        except Exception as erreur:
            print("Erreur de génération ou de préparation :", erreur)

        return resultats


class Extract_post:
    def __init__(self):
        GROQ_API_KEY = os.getenv("TOKEN_AI_DESCRYB_IMG", "")
        self.model_groq_name = "qwen/qwen3.8-27b"
        self.client = Groq(api_key=GROQ_API_KEY)
        self.TRUSTED_WEB_SOURCES = TRUSTED_WEB_SOURCES

    def clean_html(self, text):
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        return soup.get_text(separator=" ", strip=True)

    def get_publication_date(self, entry):
        published = entry.get("published_parsed")
        updated = entry.get("updated_parsed")
        date_struct = published or updated
        if not date_struct:
            return None, None
        date_utc = datetime(*date_struct[:6], tzinfo=timezone.utc)
        return date_utc.strftime("%Y-%m-%d %H:%M"), date_utc

    def fetch_all_news(self):
        articles = []
        seen_links = set()

        for source_name, feed_url in self.TRUSTED_WEB_SOURCES.items():
            try:
                feed = feedparser.parse(feed_url)
                if getattr(feed, "status", 200) >= 400:
                    continue

                for entry in feed.entries:
                    link = entry.get("link", "").strip()
                    title = entry.get("title", "").strip()

                    if not title or not link or link in seen_links:
                        continue

                    raw_text = (
                        entry.get("content", [{}])[0].get("value", "")
                        if entry.get("content")
                        else entry.get("summary", "")
                    )

                    clean_text = self.clean_html(raw_text)
                    publication_text, publication_datetime = self.get_publication_date(entry)

                    if publication_text is None:
                        continue

                    seen_links.add(link)
                    articles.append({
                        "source_name": source_name,
                        "title": self.clean_html(title),
                        "date": publication_text,
                        "continue": clean_text,
                        "_datetime": publication_datetime,
                    })
            except Exception as error:
                print(f"Erreur pour {source_name}: {error}")

        articles.sort(key=lambda item: item["_datetime"], reverse=True)
        
        result = {}
        for article in articles:
            src = article.pop("source_name")
            article.pop("_datetime")
            result[f"{src}_{article['date']}"] = article
        return result

    def tradication_of_text(self, text):
        prompt = """
        You are a professional Arabic translator specializing in financial news.
        Translate into natural Modern Standard Arabic. Preserve financial meaning and facts.
        Return ONLY a valid JSON object:
        {"caption": "translated_text_here"}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_groq_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"SOURCE TEXT TO TRANSLATE:\n{text}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_output = response.choices[0].message.content
            return json.loads(raw_output)
        except Exception as e:
            print(f"Erreur lors de l'appel Groq : {e}")
            return None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fichier_json = os.path.join(base_dir, "historique_posts.json")

    extract_post = Extract_post()
    text_post = TextPost()

    historique = {}

    if os.path.exists(fichier_json):
        try:
            with open(fichier_json, "r", encoding="utf-8") as f:
                historique = json.load(f)
            if not isinstance(historique, dict):
                historique = {}
        except Exception:
            historique = {}
    else:
        with open(fichier_json, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=4)

    fetch_all_news = extract_post.fetch_all_news()

    for source, news in fetch_all_news.items():
        vrai_source = source.split("_")[0]
        title = news["title"]
        content = news["continue"]
        date = news["date"]

        post_id = f"{vrai_source}_{title}_{date}"

        if post_id in historique:
            print(f"Déjà traité : {title}")
            continue

        text_original = f"{title}\n{content}\n\nsource: {vrai_source}, date: {date}"
        text_translated = extract_post.tradication_of_text(text_original)

        if isinstance(text_translated, dict):
            text_translated_net = " ".join(str(value) for value in text_translated.values())
        else:
            text_translated_net = str(text_translated)

        try:
            resultat = text_post.publier_texte_partout(text_original, text_translated_net)

            if resultat and (resultat.get("facebook") or resultat.get("instagram")):
                historique[post_id] = {"source": vrai_source, "titre": title, "date": date}
                with open(fichier_json, "w", encoding="utf-8") as f:
                    json.dump(historique, f, ensure_ascii=False, indent=4)
                print(f"Historique sauvegardé : {title}")
            else:
                print(f"Publication non confirmée : {title}.")
        except Exception as e:
            print(f"Erreur publication de '{title}' : {e}")
            continue
        break

    print("Fin de cette exécution.")


if __name__ == "__main__":
    main()

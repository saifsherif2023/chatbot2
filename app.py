import os
import sys
import io
import json
import requests
import logging
import traceback
import re
import random
from datetime import datetime
from collections import deque
from typing import List, Dict, Tuple, Optional, Union
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from fuzzywuzzy import fuzz, process
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from pymongo.errors import PyMongoError

load_dotenv()

app = Flask(__name__)
CORS(app)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Force stdout to UTF-8

logger = logging.getLogger(__name__)

class ChatbotError(Exception):
    """Base error class for chatbot errors"""
    pass

class ValidationError(ChatbotError):
    """Error for invalid input validation"""
    pass

class DatabaseError(ChatbotError):
    """Error for database operations"""
    pass

class EntityExtractionError(ChatbotError):
    """Error for entity extraction failures"""
    pass

class LanguageError(ChatbotError):
    """Error for language detection/processing"""
    pass

class FilterError(ChatbotError):
    """Error for filter application failures"""
    pass

class IntentClassificationError(ChatbotError):
    """Error for intent classification failures"""
    pass

class ProductQueryError(ChatbotError):
    """Error for product query failures"""
    pass

class PriceQueryError(ChatbotError):
    """Error for price query failures"""
    pass

class ServiceInitializationError(ChatbotError):
    """Error for service initialization failures"""
    pass

class ResourceCleanupError(ChatbotError):
    """Error for resource cleanup failures"""
    pass

class EmbeddingService:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2", confidence_threshold=0.25, products_collection=None):
        """Initialize the embedding service with a multilingual model and FAISS support"""
        try:
            self.model = SentenceTransformer(model_name)
            self.confidence_threshold = confidence_threshold
            self.intent_phrases = self._initialize_intent_phrases()
            if not self.intent_phrases:
                raise ValueError("Failed to initialize intent phrases")
            
            self.products_collection = products_collection
            self.context_window = deque(maxlen=5)
            self.fuzzy_threshold = 0.6
            self.product_index = None
            self.product_lookup = {}

            self._precompute_embeddings()          # For intents
            self._build_faiss_index()

        except Exception as e:
            logger.error(f"Error initializing embedding service: {str(e)}")
            raise
    
    def normalize_string(self, s: str) -> str:
        """Clean and normalize a string for comparison."""
        return re.sub(r"\s+", " ", s).strip().lower()

    def _initialize_intent_phrases(self):
        """Initialize intent phrases for different languages with expanded coverage."""
        try:
            intent_phrases = {
                'greeting': {
                    'en': [
                        'hello', 'hi', 'hey', 'hey there', 'what’s up?', 'greetings', 'good morning', 'good afternoon',
                        'good evening', 'how are you', 'howdy', 'welcome', 'hiya', 'yo', 'sup', 'hiiiii', 'heyy', 'hey bot',
                        'helloo', 'morning!', 'evenin', 'hi bot', 'hey u there?', 'what’s good?', 'nice to meet you',
                        'how’s it going?', 'what’s new?', 'long time no see', 'how are you doing?'
                    ],
                    'ar': [
                        'مرحبا', 'اهلا', 'السلام عليكم', 'كيف حالك', 'صباح الخير', 'مساء الخير',
                        'اهلا وسهلا', 'كيفك', 'هلا', 'اهلين', 'هاي', 'هلا والله', 'ازيك', 'عامل ايه',
                        'مساء النور', 'يا صباح الفل', 'ايه الاخبار', 'هاي شات', 'شو الاخبار', 'إزيك يا شات؟',
                        'وينك', 'كل سنة وانت طيب', 'إزيكم', 'أهلاً بيك', 'نهارك سعيد', 'ازيك النهاردة؟',
                        'ازيك يا باشا؟', 'كيف الحال؟', 'كل حاجة تمام؟', 'يا هلا'
                    ],
                    'mixed': [
                        'hello كيفك', 'hi مرحبا', 'hey السلام عليكم', 'welcome اهلا وسهلا', 'hi ازيك', 'صباح الخير bot',
                        'hello يا شات', 'hey ازيك', 'هلا there', 'what’s up يا باشا', 'yo عامل ايه', 'hi كيفك',
                        'مساء الخير bro', 'good morning يا جماعة', 'how are you يا شات؟', 'what’s up يا شباب'
                    ]
                },
                'recommendation': {
                    'en': [
                        'recommend something', 'suggest product', 'what should I get?', 'suggestion please',
                        'what’s popular?', 'best-sellers?', 'top picks', 'what’s trending?', 'any recommendations?',
                        'what do you suggest?', 'what’s hot right now?', 'give me ideas', 'what’s your favorite?'
                    ],
                    'ar': [
                        'رشحلي حاجة', 'اقترح حاجة', 'عايز ترشيح', 'تنصحني بايه؟', 'ايه الاشهر عندكم؟',
                        'ايه المنتجات الشعبية؟', 'شو التريند؟', 'ايه اللي مميز؟', 'ممكن ترشحلي حاجة؟',
                        'ايه المنتجات اللي بتعجب الناس؟', 'ايه الأفضل عندكم؟', 'ايه اللي يناسبني؟',
                        'ايه المنتجات اللي بتتباع كتير؟', 'عايز حاجة حلوة', 'ايه اللي تنصحني به؟'
                    ],
                    'mixed': [
                        'عايز recommendation', 'اقتراح لو سمحت', 'suggest حاجه', 'what’s popular يا شات؟',
                        'best-sellers عندكم؟', 'ايه الtrending items؟', 'recommend حاجة يدوية'
                    ]
                },
                'help': {
                    'en': [
                        'help', 'how to use', 'how does this work', 'what can you do', 'show me how', 'guide me',
                        'instructions', 'tutorial', 'how to search', 'how to browse', 'how to filter', 'help me',
                        'can you help?', 'need assistance', 'i need help', 'what can u do?', 'how do i use this?',
                        'what’s your function?', 'bot help', 'not sure what to do', 'explain please', 'how can i search?',
                        'any instructions?', 'tell me what you do', 'what’s your job?', 'stuck here', 'pls assist',
                        'can u guide me?', 'what’s this for?', 'how to get started?', 'what are my options?',
                        'how do I find products?', 'can you show me around?'
                    ],
                    'ar': [
                        'ساعدني', 'محتاج مساعدة', 'اعمل ايه؟', 'مش عارف أبدأ', 'ممكن شرح؟', 'انت بتعمل ايه؟',
                        'ايه خدماتك؟', 'محتاج اعرف اشتغل ازاي', 'عايز افهم', 'وضحلي لو سمحت', 'بتشتغل ازاي؟',
                        'ازاي استخدم البوت؟', 'انا تايه', 'مساعدة لو سمحت', 'دليني', 'بساعد في ايه؟', 'ايه قدراتك؟',
                        'مساعدة', 'كيفية الاستخدام', 'كيف يعمل', 'ماذا يمكنك ان تفعل', 'اشرح لي', 'دلني', 'تعليمات',
                        'شرح', 'كيفية البحث', 'كيفية التصفح', 'كيفية التصفية', 'وريني إزاي أستخدم البوت',
                        'إزاي أفلتر المنتجات؟', 'كيف أبحث عن حاجة؟', 'علمني استخدم الشات', 'ازاي اقدر اشوف المنتجات؟',
                        'ازاي ألاقي المنتجات؟', 'ايه الخيارات المتاحة؟', 'ازاي أبدأ؟', 'ممكن تساعدني ألاقي حاجة؟'
                    ],
                    'mixed': [
                        'help محتاج', 'can u تساعدني؟', 'شو بتعمل bot?', 'ايه help options؟', 'i need مساعده',
                        'how u work يا بوت؟', 'عايز help', 'explain بالعربي', 'help شرح', 'كيفية الاستخدام guide',
                        'how to use شرح', 'ماذا يمكنك ان تفعل tutorial', 'كيف أبحث عن products؟', 'how to filter المنتجات'
                    ]
                },
                'feedback': {
                    'en': [
                        'this is great', 'not helpful', 'i like this', 'bad response', 'awesome!', 'you helped me', 'didn’t like that', 'thanks!',
                        'good job', 'you’re dumb', 'love it', 'boring answer', 'that worked', 'didn’t work', 'helped a lot', 'useless', 'nice',
                        'terrible', 'cool response', 'meh', 'feedback', 'suggestion', 'complaint', 'report issue', 'tell you something', 
                        'let you know', 'share my thoughts', 'what I think', 'my opinion', 'review', 'that was helpful', 'not what I expected',
                        'could be better', 'amazing!', 'I appreciate it', 'not bad', 'could use improvement'
                    ],
                    'ar': [
                        'شكراً', 'كويس', 'مش عاجبني', 'حلو الكلام', 'رد وحش', 'تمام كده', 'مش مفيد', 'برافو', 'مساعدتش', 'كلام فاضي',
                        'عجبني', 'رد ممتاز', 'سيء', 'ردك مش واضح', 'مش واضح', 'شكراً على المساعدة', 'استفدت', 'حبيت الرد', 'ردك حلو', 'موفق',
                        'تقييم', 'اقتراح', 'شكوى', 'ابلاغ عن مشكلة', 'اقولك حاجة', 'اعرفك', 'شارك افكاري', 'رأيي', 'تقييمي', 'مراجعة',
                        'كان مفيد', 'مش اللي كنت متوقعه', 'ممكن يكون أحسن', 'ممتاز!', 'أنا أقدر ده', 'مش وحش', 'محتاج تحسين'
                    ],
                    'mixed': [
                        'جميل response', 'مو حلو reply', 'thank you يا بوت', 'ما فهمتش the answer', 'not bad يا مان', 'awesome شكراً', 
                        'ردك was great', 'feedback تقييم', 'suggestion اقتراح', 'review تقييمي', 'share my thoughts شارك افكاري',
                        'كان helpful', 'مش اللي كنت عايزه', 'could be better يا شات', 'ردك حلو قوي'
                    ]
                },
                'filter': {
                    'en': [
                        'filter by', 'show only', 'sort by', 'arrange by', 'organize by', 'group by', 'categorize by',
                        'price range', 'rating', 'location', 'artisan', 'color', 'size', 'material',
                        'filter by color', 'show red items', 'only crochet', 'i want leather', 'filter category pottery',
                        'bags only', 'do you have blue?', 'search by artisan', 'filter by size', 'wood products',
                        'cotton stuff please', 'show me local items', 'egypt-made items', 'organic only',
                        'i want eco stuff', 'natural fiber', 'small size', 'show soft material', 'filter by handmade type',
                        'filter by price', 'show items under 100', 'only high-rated products', 'filter by category'
                    ],
                    'ar': [
                        'تصفية حسب', 'عرض فقط', 'ترتيب حسب', 'تنظيم حسب', 'تجميع حسب', 'تصنيف حسب', 'فئة',
                        'نطاق السعر', 'التقييم', 'الموقع', 'الحرفي', 'اللون', 'الحجم', 'المادة',
                        'فلتر حسب اللون', 'عايز شنط جلد', 'منتجات كروشيه بس', 'حاجات لونها أحمر', 'حاجات من الخشب',
                        'خامة طبيعية', 'فلتر بالحجم', 'من صنع يدوي', 'بس صناعة مصرية', 'فلتر باللون الأزرق',
                        'شنط بس', 'مواد عضوية', 'منتجات من الصوف', 'الفلتر حسب الفئة', 'الأعمال المحلية',
                        'حاجات يدوية فقط', 'عايز فخار', 'فلتر بالمقاس', 'محتاج حاجه جلد', 'بس الحاجات القطن',
                        'وريني شنط جلد', 'عندك حاجه لونها أحمر؟', 'عايز منتجات صغيرة', 'حاجات من الخشب',
                        'في شنط لونها أسود؟', 'منتجات لونها أخضر', 'حاجه بتتلبس', 'منتجات من الجلد',
                        'حاجات خفيفة', 'منتجات حجمها كبير', 'حاجه من القطن', 'فلتر بالسعر', 'وريني حاجات تحت ١٠٠',
                        'بس المنتجات عالية التقييم', 'فلتر حسب الفئة'
                    ],
                    'mixed': [
                        'filter by تصفية', 'sort by ترتيب', 'group by تجميع', 'price range نطاق السعر',
                        'فلتر category bags', 'حاجه soft material', 'show me الفخار', 'عايز crochet items',
                        'بس items جلد', 'display شنط جلد', 'filter by الحجم', 'filter by price يا شات',
                        'show only عالي التقييم', 'filter category pottery'
                    ]
                },
                'product_query': {
                    'en': [
                        'show me products', 'what do you sell?', 'looking for handmade items', 'i want to buy something', 'do you have jewelry?', 
                        'any accessories?', 'i need a bag', 'what’s available?', 'got any gifts?', 'i saw a product on insta', 'need a necklace', 
                        'any nice handmade stuff?', 'search products', 'what’s in store?', 'products please', 'show handmade options', 'i’m shopping', 
                        'looking for ideas', 'got pottery?', 'what products do you have', 'list products', 'show me items', 'what items are available', 
                        'display products', 'browse products', 'view products', 'show me everything', 'what can I buy?', 'do you have any new items?'
                    ],
                    'ar': [
                        'وريني المنتجات', 'ايه عندكم؟', 'منتجات متاحة', 'حابب اشتري حاجه', 'عندك اكسسوارات؟', 'محتاج حاجة يدوي', 'ايه اللي موجود؟', 
                        'فين السلع؟', 'ببيعوا ايه؟', 'عايز خاتم', 'شنط عندكم؟', 'حرف يدوية عندك؟', 'منتج يدوي', 'فين الاباجورات؟', 'ابحث عن منتجات', 
                        'دلني على منتجات', 'منتجات رمضان', 'معندكش فانوس؟', 'فين الطاقم الخرز؟', 'عرض المنتجات', 'ما هي المنتجات المتوفرة', 
                        'قائمة المنتجات', 'عرض السلع', 'ما هي السلع المتوفرة', 'عرض البضائع', 'تصفح المنتجات', 'البحث عن منتجات', 'عرض الاصناف',
                        'اعرضلي كل المنتجات', 'عايز أشوف المنتجات', 'عندك حاجات يدوية؟', 'ايه المنتجات اللي عندكم؟', 'عندك فانوس؟', 'عايز منتجات', 'ممكن أشوف المنتجات؟',
                        'في عندكم ايه؟', 'وريني الحاجات اللي عندكم', 'عايز اشوف كل حاجة', 'ايه الجديد عندكم؟', 'عندك حاجات جديدة؟'
                    ],
                    'mixed': [
                        'عايز handmade stuff', 'do u have فخار؟', 'شنطة يد please', 'any bracelets يدوية؟', 'منتجات من wood؟', 'need something يدوي',
                        'show me المنتجات', 'عرض products', 'what المنتجات do you have', 'list المنتجات المتوفرة', 'show me السلع',
                        'looking for اكسسوارات', 'عايز اشوف handmade items', 'what’s available يا شات؟'
                    ]
                },
                'price_query': {
                    'en': [
                        'show me products under', 'products less than', 'items below', 'products over', 'items more than', 'products above',
                        'products between', 'items from to', 'price range', 'show me cheap products', 'show me expensive products',
                        'show me cheap items', 'under 100 please', 'any discounts?', 'what’s the price?', 'is it expensive?',
                        'how much?', 'looking for budget items', 'affordable things', 'less than 200', 'need discount', 'cost?',
                        'show sales', 'i’m on a budget', 'low price options', 'anything cheaper?', 'price below 150', 'cut the cost', 'sale products?',
                        'what’s the price range?', 'do you have deals?', 'any promotions?', 'what’s on sale?'
                    ],
                    'ar': [
                        'عرض منتجات اقل من', 'منتجات تحت', 'سلع اقل من', 'منتجات اكثر من', 'سلع فوق', 'منتجات بين', 'منتجات من الى', 'نطاق السعر',
                        'عرض منتجات رخيصة', 'عرض منتجات غالية', 'وريني الأرخص', 'أقل من ٢٠٠', 'حاجه بسعر كويس', 'معندكش خصومات؟', 'بكام دي؟',
                        'سعرها ايه؟', 'أنا على قد budget', 'بكام الحاجة دي؟', 'منتجات رخيصة', 'كم سعرها؟', 'الأسعار كام؟', 'محتاج حاجه بـ ١٠٠',
                        'أقل سعر', 'حاجات عليها خصم', 'في عروض؟', 'المنتجات المخفضة', 'محتاج حاجه مش غالية', 'منتجات تحت ٥٠٠', 'ارخص حاجة ايه؟',
                        'منتجات تحت ٢٠٠', 'الحاجات اللي فوق ٥٠٠ جنيه', 'عندك خصومات؟', 'منتجات رخيصة', 'حاجه تحت ١٠٠ جنيه', 'ايه الحاجات المخفضة؟', 'منتجات بين ٥٠ و ٢٠٠',
                        'ايه نطاق السعر؟', 'في عروض؟', 'ايه اللي على الخصم؟', 'في حاجات مخفضة؟'
                    ],
                    'mixed': [
                        'show me منتجات under', 'عرض products اقل من', 'products بين', 'منتجات between',
                        'products تحت 200', 'ايه ارخص item؟', 'need حاجة cheap', 'خصومات available؟',
                        'عرض موجود؟', 'كم الprice؟', 'budget products لو سمحت', 'what’s the price يا شات؟',
                        'any discounts عندكم؟', 'show me cheap حاجات'
                    ]
                },
                'category_query': {
                    'en': [
                        'show me jewelry', 'bags category', 'looking for pottery', 'do you sell accessories?', 'anything under home decor?', 
                        'necklaces section', 'bracelets category', 'filter to fashion', 'i want candles', 'crafts for kitchen', 'display vases', 
                        'only decor', 'earrings category', 'gift items?', 'art category', 'cultural crafts?', 'wall decor?', 'boho stuff', 
                        'anything under wood?', 'show me categories', 'list categories', 'what categories do you have', 'browse categories',
                        'view categories', 'search by category', 'filter by category', 'show products in category', 'what’s in the pottery section?',
                        'do you have a section for handmade items?', 'show me all decor items', 'what categories are available?'
                    ],
                    'ar': [
                        'فئة الحلي', 'شنط يد', 'وريني قسم الفخار', 'اكسسوارات', 'حرف للمطبخ', 'مصنوعات خشبية', 'مجموعة الخواتم', 'قسم الديكور', 
                        'عايز شموع', 'فنون يدوية', 'منتجات ثقافية', 'الأباجورات', 'الهدايا', 'حرف رمضانية', 'القسم الفني', 'تحت الحرف اليدوية', 
                        'عايز اطباق', 'فخار بس', 'فئة الكروشية', 'عرض التصنيفات', 'قائمة التصنيفات', 'ما هي التصنيفات المتوفرة',
                        'تصفح التصنيفات', 'البحث حسب التصنيف', 'تصفية حسب التصنيف', 'عرض منتجات في التصنيف', 'وريني قسم الجلد',
                        'فئة الشنط', 'حاجات ديكور', 'منتجات فخار', 'قسم الأكسسوارات', 'ايه في قسم الفخار؟', 'عندك قسم للحاجات اليدوية؟',
                        'وريني كل حاجات الديكور', 'ايه التصنيفات المتاحة؟'
                    ],
                    'mixed': [
                        'عرض category الشنط', 'products from فئة decor', 'عايز craft items', 'شنط in accessories', 'فخار category only', 
                        'candle قسم', 'section فني', 'show me التصنيفات', 'عرض categories', 'what التصنيفات do you have',
                        'list التصنيفات المتوفرة', 'looking for pottery قسم', 'show me handmade items فئة', 'what’s in decor category يا شات؟'
                    ]
                },
                'next_page': {
                    'en': [
                        'show more', 'next page', 'load more', 'more products', 'next items', 'show next', 'continue browsing',
                        'next', 'more please', 'load more products', 'what else?', 'keep going', 'any more?', 'next set',
                        'more items?', 'keep browsing', 'scroll down', 'give me more', 'continue', 'what’s after that?', 'next list',
                        'see more', 'load additional items', 'more options', 'show me the rest'
                    ],
                    'ar': [
                        'عرض المزيد', 'الصفحة التالية', 'تحميل المزيد', 'منتجات اكثر', 'سلع اكثر', 'عرض التالي', 'متابعة التصفح',
                        'التالي', 'كمل', 'وريني كمان', 'المزيد من المنتجات', 'عايز اشوف اكتر', 'في كمان؟', 'كمللي الباقي',
                        'عايز اكتر', 'عرض تاني', 'اللي بعده', 'التالي من القائمة', 'كمل بحث', 'فين الباقي؟', 'باقي العناصر', 'كمل يا بوت',
                        'وريني الباقي', 'حمل حاجات زيادة', 'خيارات اكتر', 'وريني الباقي'
                    ],
                    'mixed': [
                        'show المزيد', 'next صفحة', 'load المزيد', 'more منتجات', 'عرض next',
                        'next منتجات', 'عرض more items', 'عايز next page', 'كمل show', 'scroll منتجات', 'load منتجات تانية', 'فين rest؟',
                        'see more حاجات', 'load additional منتجات', 'show me more يا شات'
                    ]
                }
            }

            logger.info(f"Initialized {len(intent_phrases)} intents with expanded multilingual phrases.")
            return intent_phrases

        except Exception as e:
            logger.error(f"Error initializing intent phrases: {str(e)}")
            raise

    def refresh_intent_index(self):
        """Reload intent phrases and rebuild FAISS index"""
        try:
            self.intent_phrases = self._initialize_intent_phrases()
            self._precompute_embeddings()
            logger.info("Successfully refreshed intent index.")
        except Exception as e:
            logger.error(f"Failed to refresh intent index: {str(e)}")


    def _precompute_embeddings(self):
        """Precompute embeddings for all intent phrases"""
        try:
            self.phrase_embeddings = {}
            for intent, phrases in self.intent_phrases.items():
                if isinstance(phrases, str):
                    phrases = [phrases]
                elif isinstance(phrases, dict):
                    phrases = [p for lang_phrases in phrases.values() for p in lang_phrases]
                
                embeddings = self.model.encode(phrases, convert_to_tensor=True)
                self.phrase_embeddings[intent] = {
                    'phrases': phrases,
                    'embeddings': embeddings
                }
            logger.info("Successfully precomputed embeddings for all intent phrases")
        except Exception as e:
            logger.error(f"Error precomputing embeddings: {str(e)}")
            raise

    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent embeddings."""
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        # Keep alphanumeric characters and Arabic characters, remove others
        text = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF\s]', '', text)
        return text
    
    def _get_category_name(self, category):
        """Resolve category name from ObjectId or dict."""
        if not category:
            return ""
        if isinstance(category, ObjectId):
            doc = self.products_collection["categories"].find_one({"_id": category}, {"name": 1})
            return doc["name"] if doc and "name" in doc else ""
        return category.get("name", "") if isinstance(category, dict) else str(category)

    def _get_subcategory_names(self, subcategories):
        """Resolve subcategory names from list of ObjectIds or strings."""
        if not subcategories:
            return []
        if isinstance(subcategories, list) and subcategories and isinstance(subcategories[0], ObjectId):
            docs = self.products_collection.database["subcategories"].find(
                {"_id": {"$in": subcategories}}, {"name": 1}
            )
            return [doc["name"] for doc in docs if "name" in doc]
        return [str(sub) for sub in subcategories if sub] if isinstance(subcategories, list) else []

    def _get_artisan_name(self, artisan):
        """Resolve artisan name from ObjectId or dict."""
        if not artisan:
            return ""
        if isinstance(artisan, ObjectId):
            doc = self.products_collection.database["users"].find_one({"_id": artisan}, {"name": 1})
            return doc["name"] if doc and "name" in doc else ""
        return artisan.get("name", "") if isinstance(artisan, dict) else str(artisan)

    def _build_faiss_index(self):
        """Build FAISS index with normalized and comprehensive product embeddings."""
        try:
            products = list(self.products_collection.find({}))
            if not products:
                logger.warning("No products found — skipping FAISS index build.")
                self.product_index = None
                self.product_lookup = {}
                return

            texts = []
            indexed_products = []

            for p in products:
                # Normalize size & weight
                normalized_size = self.normalize_size(p.get("size", ""))
                normalized_weight = self.normalize_weight(p.get("weight", ""))

                # Get category name
                category_name = self._get_category_name(p.get("category"))

                # Get subcategory names
                subcategory_names = self._get_subcategory_names(p.get("subcategories", []))

                # Normalize colors
                colors = p.get("colors", [])
                if not isinstance(colors, list):
                    colors = [colors] if colors else []

                # Get artisan name
                artisan_name = self._get_artisan_name(p.get("artisan"))

                # Include additional fields
                material = p.get("material", "")
                location = p.get("location", "")

                # Build text for embedding with normalization
                text = f"{self.normalize_text(p.get('title', ''))} {self.normalize_text(p.get('description', ''))} " \
                    f"{self.normalize_text(category_name)} {self.normalize_text(artisan_name)} " \
                    f"{' '.join([self.normalize_text(sc) for sc in subcategory_names])} " \
                    f"{' '.join([self.normalize_text(c) for c in colors])} " \
                    f"{normalized_size} {normalized_weight} " \
                    f"{self.normalize_text(material)} {self.normalize_text(location)}"

                cleaned = re.sub(r"\s+", " ", text).lower().strip()
                if cleaned:
                    texts.append(cleaned)
                    p["size"] = normalized_size
                    p["weight"] = normalized_weight
                    indexed_products.append(p)
                else:
                    logger.debug(f"Skipping product {p.get('title', 'unknown')} due to empty text")

            if not texts:
                logger.warning("No valid product texts — skipping FAISS index build.")
                self.product_index = None
                self.product_lookup = {}
                return

            embeddings = self.model.encode(texts, convert_to_numpy=True)
            dimension = embeddings.shape[1]
            self.product_index = faiss.IndexFlatL2(dimension)
            self.product_index.add(embeddings)
            self.product_lookup = {i: product for i, product in enumerate(indexed_products)}

            logger.info(f"FAISS index built with {len(self.product_lookup)} products.")

        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            self.product_index = None
            self.product_lookup = {}


    def initialize_product_index(self, products: List[Dict]):
        """Initialize FAISS index for product search"""
        # Create product embeddings
        product_texts = []
        for product in products:
            # Combine relevant product fields
            text = f"{product.get('title', '')} {product.get('description', '')} "
            text += f"{product.get('category', '')} {product.get('colors', '')} "
            text += f"{product.get('artisan', '')} {product.get('location', '')} "
            text += f"{product.get('price', '')} {product.get('rating', '')}"
            product_texts.append(text)
            
        embeddings = self.model.encode(product_texts, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        
        dimension = embeddings_np.shape[1]
        self.product_index = faiss.IndexFlatL2(dimension)
        self.product_index.add(embeddings_np)
        self.product_metadata = products


    def search_products(self, query: str, k: int = 5) -> List[Dict]:
        """Search products using semantic similarity via FAISS, with robust fallback signaling."""
        try:
            if not hasattr(self, "product_index") or self.product_index is None:
                logger.warning("FAISS product index not initialized. Skipping search.")
                return [{"_faiss_error": True}]

            if not hasattr(self, "product_lookup") or not self.product_lookup:
                logger.warning("Product lookup dictionary not available or empty. Skipping search.")
                return [{"_faiss_error": True}]

            query_embedding = self.model.encode([query])
            query_embedding_np = np.array(query_embedding).astype('float32')

            if query_embedding_np.ndim != 2:
                logger.warning("FAISS query embedding is not 2D, skipping search.")
                return [{"_faiss_error": True}]

            _, indices = self.product_index.search(query_embedding_np, k)

            results = []
            for idx in indices[0]:
                if idx != -1 and idx in self.product_lookup:
                    results.append(self.product_lookup[idx])

            return results or []  # Return empty if FAISS worked but found nothing

        except Exception as e:
            logger.error(f"Error during FAISS product search: {str(e)}")
            return [{"_faiss_error": True}]

    def detect_intent(self, text: str, lang: str) -> Tuple[str, float]:
        """Detect intent from text using multilingual embeddings with context and fallbacks.
        Assumes text is preprocessed (normalized and transliterated) prior to calling."""
        try:
            text_embedding = self.model.encode(text, convert_to_tensor=True)
            
            # Check for English and Arabic characters
            has_english = any(c.isascii() for c in text)
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
            is_mixed = has_english and has_arabic
            
            best_intent = None
            best_score = 0.0
            
            # Determine patterns to try based on language detection
            patterns_to_try = ['ar', 'en', 'mixed'] if is_mixed else ['ar' if has_arabic else 'en']
            
            # Compare text embedding with intent phrase embeddings
            for pattern_type in patterns_to_try:
                for intent, data in self.phrase_embeddings.items():
                    if pattern_type in self.intent_phrases[intent]:
                        phrases = self.intent_phrases[intent][pattern_type]
                        embeddings = self.model.encode(phrases, convert_to_tensor=True)
                        similarities = torch.nn.functional.cosine_similarity(
                            text_embedding.unsqueeze(0),
                            embeddings
                        )
                        max_score = float(torch.max(similarities))
                        
                        # Boost mixed-language scores slightly
                        if pattern_type == 'mixed':
                            max_score *= 1.05
                            
                        # Apply context boost if context exists
                        if len(self.context_window) > 0:
                            context_boost = self._get_context_boost(intent)
                            max_score *= context_boost
                            
                        if max_score > best_score:
                            best_score = max_score
                            best_intent = intent
            
            # Fallback to fuzzy matching if confidence is low
            if best_score < self.confidence_threshold:
                fuzzy_intent, fuzzy_score = self._fuzzy_match_intent(text, lang)
                fuzzy_threshold = 0.5 if lang == 'ar' else 0.6  # Lower threshold for Arabic
                if fuzzy_score > fuzzy_threshold and fuzzy_score > best_score:
                    best_intent = fuzzy_intent
                    best_score = fuzzy_score
            
            # Fallback to context prediction if still low confidence
            if best_score < self.confidence_threshold and len(self.context_window) > 0:
                context_intent = self._predict_from_context()
                if context_intent:
                    best_intent = context_intent
                    best_score = 0.3
            
            # Update context window
            self.context_window.append((text, best_intent or "unknown", best_score))
            
            return best_intent or "unknown", best_score
            
        except Exception as e:
            logger.error(f"Error in intent detection: {str(e)}")
            return "unknown", 0.0

    def normalize_size(self, value: str) -> str:
        try:
            value = value.lower().strip()
            if any(unit in value for unit in ["cm", "x", "*", "×", "mm"]):
                numbers = re.findall(r"[\d\.]+", value)
                nums = [float(n) for n in numbers]
                if len(nums) >= 2:
                    area = nums[0] * nums[1]
                elif nums:
                    area = nums[0]
                else:
                    return "unknown"
                if area <= 100:
                    return "small"
                elif area <= 400:
                    return "medium"
                else:
                    return "large"
            elif "small" in value:
                return "small"
            elif "medium" in value:
                return "medium"
            elif "large" in value:
                return "large"
        except Exception as e:
            logger.debug(f"Size normalization failed for '{value}': {str(e)}")
        return "unknown"

    def normalize_weight(self, value: str) -> str:
        """Normalize weight to a category (light, medium, heavy) based on value and unit"""
        try:
            if value is None or value == "":
                return "unknown"
            
            # Handle numeric inputs (assume kilograms)
            if isinstance(value, (int, float)):
                num = float(value)
                if num < 1:
                    return "light"
                elif num <= 3:
                    return "medium"
                else:
                    return "heavy"
            
            # Handle string inputs
            if isinstance(value, str):
                value = value.lower().strip()
                # Extract number and unit (e.g., '1.5 kg', '500g')
                match = re.match(r'(\d*\.?\d+)\s*(kg|g|kilogram|gram)?', value)
                if not match:
                    logger.debug(f"Invalid weight format: '{value}'")
                    return "unknown"
                
                num = float(match.group(1))
                unit = match.group(2) or "kg"  # Default to kg if no unit
                if unit in ["g", "gram"]:
                    if num < 500:
                        return "light"
                    elif num <= 1500:
                        return "medium"
                    else:
                        return "heavy"
                elif unit in ["kg", "kilogram"]:
                    if num < 1:
                        return "light"
                    elif num <= 3:
                        return "medium"
                    else:
                        return "heavy"
            
            # Handle unexpected types
            logger.debug(f"Invalid weight type: {type(value)} for value: {value}")
            return "unknown"
        
        except Exception as e:
            logger.debug(f"Weight normalization error for '{value}': {str(e)}")
            return "unknown"
    
    def _get_context_boost(self, intent: str) -> float:
        """Calculate context boost based on previous messages using stored intents"""
        if len(self.context_window) == 0:
            return 1.0
            
        intent_count = sum(1 for _, stored_intent, _ in self.context_window if stored_intent == intent)
                
        if intent_count > 0:
            return 1.0 + (0.1 * intent_count)
        return 1.0


    def _fuzzy_match_intent(self, text: str, lang: str) -> Tuple[str, float]:
        """Fallback to fuzzy matching for low confidence cases"""
        best_intent = None
        best_score = 0.0
        
        for intent, phrases in self.intent_phrases.items():
            if lang in phrases:
                for phrase in phrases[lang]:
                    score = fuzz.ratio(text.lower(), phrase.lower()) / 100.0
                    if score > best_score:
                        best_score = score
                        best_intent = intent
                        
        return best_intent or "unknown", best_score


    def _predict_from_context(self) -> Optional[str]:
        """Predict intent based on conversation context using stored intents"""
        if len(self.context_window) == 0:
            return None
            
        intent_counts = {}
        for _, stored_intent, _ in self.context_window:
            intent_counts[stored_intent] = intent_counts.get(stored_intent, 0) + 1
                
        if intent_counts:
            most_common = max(intent_counts.items(), key=lambda x: x[1])
            if most_common[1] >= 2:
                return most_common[0]
                
        return None

class ChatbotService:
    def __init__(self, mongo_uri=None, db_name="handMade", recommendation_service_url=None):
        """Initialize chatbot service with enhanced error handling"""
        try:
            if not mongo_uri:
                raise ServiceInitializationError("MongoDB URI is required")
                
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.products_collection = self.db.products
            self.categories_collection = self.db.categories
            self.subcategories_collection = self.db.subcategories
            self.users_collection = self.db.users

            self.responses = {
                "en": {
                    "greeting": ["Hello! How can I help you today?"],
                    "help": ["I'm here to help! You can ask me to show products, filter by price, category, and more."],
                    "fallback": ["I'm not sure I understood that. Could you rephrase?"],
                    "clarify": ["Could you please clarify your request?"],
                    "clarify_category": ["What kind of product are you looking for?"],
                    "clarify_price": ["Do you have a budget in mind?"],
                    "conflict": ["I found conflicting information in your request."],
                    "conflicting_price": ["You mentioned both cheap and expensive items. Which one do you prefer?"],
                    "price_range_products": ["Here are some products between {min_price} and {max_price} {currency}:"],
                    "no_products_price": ["I couldn't find products in the {min_price}–{max_price} {currency} range."],
                    "rating_results": ["Here are products rated {rating}+ stars:"],
                    "no_results_rating": ["No products found with ratings above that level."],
                    "popular": ["These are some of our most popular handmade items:"],
                    "faiss_down": [
                        "Sorry, our smart product suggestions are currently unavailable. Please try again later or use filters."
                    ],
                    "product_query": ["Here are some handmade products you might like:"],
                    "filter": {
                        "success": "Here are {count} items matching your filters:",
                        "no_results": "I couldn’t find any products with those filters. Try changing them?"
                    },
                    "no_results": ["I couldn't find any matching products. Try adjusting your filters or search terms?"],
                    "error": ["Something went wrong. Please try again later."],
                    "invalid_price_range": ["The price range you entered seems incorrect."],
                    "price_range_example": [
                        "Try saying: Show me products between 100 and 300 EGP",
                        "Or: I want items under 200 pounds"
                    ],
                    "price_clarification": [
                        "Do you want affordable or luxury products?",
                        "Try: Show me cheap items or Show me expensive products"
                    ],
                    "invalid_rating": ["Ratings must be between 0 and 5 stars."],
                    "rating_example": [
                        "Try: Show me products rated above 4",
                        "Or: I want items with a rating of 3.5 or more"
                    ],
                    "error_messages": {
                        "ValidationError": "I couldn't understand that. Could you please rephrase?",
                        "DatabaseError": "I'm having trouble accessing the database. Please try again later.",
                        "EntityExtractionError": "I had trouble understanding some details. Could you be more specific?",
                        "LanguageError": "I'm having trouble with the language. Could you try in English or Arabic?",
                        "FilterError": "I couldn't apply those filters. Could you try different criteria?",
                        "IntentClassificationError": "I'm not sure what you're looking for. Could you rephrase your request?",
                        "ProductQueryError": "I had trouble finding products. Could you try a different search?",
                        "PriceQueryError": "I couldn't process the price range. Could you specify a different range?",
                        "ServiceInitializationError": "The service is not properly initialized. Please try again later.",
                        "ResourceCleanupError": "There was an issue cleaning up resources. The service will continue to work.",
                        "default": "I encountered an error. Please try again."
                    }
                },
                "ar": {
                    "greeting": ["مرحبًا! كيف يمكنني مساعدتك؟"],
                    "help": ["أنا هنا للمساعدة! يمكنك أن تطلب مني عرض منتجات، أو تصفية حسب السعر أو الفئة، والمزيد."],
                    "fallback": ["لم أفهم ذلك. هل يمكنك إعادة الصياغة؟"],
                    "clarify": ["هل يمكنك توضيح طلبك أكثر؟"],
                    "clarify_category": ["ما نوع المنتج الذي تبحث عنه؟"],
                    "clarify_price": ["هل لديك ميزانية معينة في بالك؟"],
                    "conflict": ["وجدت معلومات متضاربة في طلبك."],
                    "conflicting_price": ["ذكرت منتجات رخيصة وغالية معًا. تفضل أي نوع؟"],
                    "price_range_products": ["إليك بعض المنتجات بين {min_price} و {max_price} {currency}:"],
                    "no_products_price": ["لم أجد منتجات في نطاق السعر {min_price}–{max_price} {currency}."],
                    "rating_results": ["إليك منتجات بتقييم {rating}+ نجوم:"],
                    "no_results_rating": ["لا توجد منتجات بتلك التقييمات حاليًا."],
                    "popular": ["هذه بعض من أكثر منتجاتنا اليدوية شهرة:"],
                    "faiss_down": [
                        "عذرًا، اقتراحات المنتجات الذكية غير متاحة حاليًا. يرجى المحاولة لاحقًا أو استخدام الفلاتر."
                    ],
                    "product_query": ["إليك بعض المنتجات اليدوية التي قد تعجبك:"],
                    "filter": {
                        "success": "تم العثور على {count} منتجًا يطابق الفلاتر:",
                        "no_results": "لم أجد منتجات بهذه الفلاتر. جرّب تغييرها؟"
                    },
                    "no_results": ["لم أتمكن من العثور على منتجات مطابقة. حاول تغيير الفلاتر أو مصطلحات البحث؟"],
                    "error": ["حدث خطأ ما. يرجى المحاولة لاحقًا."],
                    "invalid_price_range": ["نطاق السعر الذي أدخلته غير صحيح."],
                    "price_range_example": [
                        "مثال: وريني منتجات بين ١٠٠ و ٣٠٠ جنيه",
                        "أو: عايز حاجه تحت ٢٠٠ جنيه"
                    ],
                    "price_clarification": [
                        "هل تفضل منتجات رخيصة أم فاخرة؟",
                        "مثال: وريني حاجات رخيصة أو منتجات غالية"
                    ],
                    "invalid_rating": ["يجب أن يكون التقييم بين 0 و 5 نجوم."],
                    "rating_example": [
                        "مثال: وريني منتجات تقييمها فوق ٤",
                        "أو: عايز حاجه تقييمها ٣.٥ أو أكثر"
                    ],
                    "error_messages": {
                        "ValidationError": "لم أفهم ذلك. هل يمكنك إعادة الصياغة؟",
                        "DatabaseError": "أواجه مشكلة في الوصول إلى قاعدة البيانات. يرجى المحاولة مرة أخرى لاحقًا.",
                        "EntityExtractionError": "واجهت صعوبة في فهم بعض التفاصيل. هل يمكنك أن تكون أكثر تحديدًا؟",
                        "LanguageError": "أواجه مشكلة في اللغة. هل يمكنك المحاولة باللغة الإنجليزية أو العربية؟",
                        "FilterError": "لم أتمكن من تطبيق هذه التصفية. هل يمكنك تجربة معايير مختلفة؟",
                        "IntentClassificationError": "لست متأكدًا مما تبحث عنه. هل يمكنك إعادة صياغة طلبك؟",
                        "ProductQueryError": "واجهت صعوبة في العثور على المنتجات. هل يمكنك تجربة بحث مختلف؟",
                        "PriceQueryError": "لم أتمكن من معالجة نطاق السعر. هل يمكنك تحديد نطاق مختلف؟",
                        "ServiceInitializationError": "الخدمة غير مهيأة بشكل صحيح. يرجى المحاولة مرة أخرى لاحقًا.",
                        "ResourceCleanupError": "كانت هناك مشكلة في تنظيف الموارد. ستستمر الخدمة في العمل.",
                        "default": "واجهت خطأ. يرجى المحاولة مرة أخرى."
                    }
                }
            }
            self.context = {}
            
            try:
                self.embedding_service = EmbeddingService(products_collection=self.products_collection)
                self.product_lookup = self.embedding_service.product_lookup  # Add this
            except Exception as e:
                raise ServiceInitializationError(f"Failed to initialize embedding service: {str(e)}")
            
            try:
                self._initialize_product_index()
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS product index: {str(e)}")

            self.recommendation_service_url = recommendation_service_url    
            logger.info("ChatbotService initialized successfully")
            
        except Exception as e:
            if isinstance(e, ServiceInitializationError):
                raise
            raise ServiceInitializationError(f"Failed to initialize ChatbotService: {str(e)}")

    def __del__(self):
        """Cleanup resources when the service is destroyed"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
            if hasattr(self, 'embedding_service'):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                del self.embedding_service.model
            logger.info("ChatbotService resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise ResourceCleanupError(f"Failed to cleanup resources: {str(e)}")

    def _extract_entities(self, text: str, lang: str) -> Dict:
        """Extract entities from text with enhanced Arabic support."""
        entities = {
            "descriptions": [],
            "categories": [],
            "subcategories": [],
            "price_range": None,
            "rating": None,
            "locations": [],
            "artisans": [],
            "colors": [],
            "size": [],
            "weights": [],
            "product_titles": []
        }

        logger.debug(f"Extracting entities from text: '{text}' (lang: {lang})")

        def safe_distinct_str(field):
            return [str(v).strip().lower() for v in self.products_collection.distinct(field) if v]

        def safe_distinct(field):
            return [v for v in self.products_collection.distinct(field) if isinstance(v, (str, float, int))]

        def normalize_arabic(text):
            """Normalize Arabic text: remove diacritics, normalize numerals."""
            arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
            text = re.sub(arabic_diacritics, '', text)
            arabic_numerals = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            return text.translate(arabic_numerals)

        try:
            text_lower = normalize_arabic(text.lower()) if lang == 'ar' else text.lower()

            # --- Product Titles (Exact Match) ---
            product_titles = self.products_collection.distinct("title")
            for title in product_titles:
                if isinstance(title, str) and normalize_arabic(title.lower()) in text_lower:
                    entities["product_titles"].append(title)
            logger.debug(f"Extracted product_titles (exact match): {entities['product_titles']}")

            # --- Price Range ---
            price_patterns = {
                'en': [
                    r'under\s+(\d+)\s*(?:pounds|egp)',
                    r'below\s+(\d+)\s*(?:pounds|egp)',
                    r'less\s+than\s+(\d+)\s*(?:pounds|egp)',
                    r'(\d+)\s*to\s*(\d+)\s*(?:pounds|egp)'
                ],
                'ar': [
                    r'تحت\s+([\d٠١٢٣٤٥٦٧٨٩]+)\s*(?:جنيه|جنيها?)',
                    r'أقل\s+من\s+([\d٠١٢٣٤٥٦٧٨٩]+)\s*(?:جنيه|جنيها?)',
                    r'من\s+([\d٠١٢٣٤٥٦٧٨٩]+)\s*إلى\s*([\d٠١٢٣٤٥٦٧٨٩]+)\s*(?:جنيه|جنيها?)',
                    r'رخيص',  # Maps to 0–200 EGP
                    r'غالي'   # Maps to 500+ EGP
                ]
            }

            for pattern in price_patterns.get(lang, []):
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        if pattern in ['رخيص', 'غالي']:
                            entities["price_range"] = [0, 200] if pattern == 'رخيص' else [500, float('inf')]
                        elif match.lastindex == 2:
                            min_price, max_price = float(normalize_arabic(match.group(1))), float(normalize_arabic(match.group(2)))
                            if min_price <= max_price:
                                entities["price_range"] = [min_price, max_price]
                        else:
                            entities["price_range"] = [0, float(normalize_arabic(match.group(1)))]
                        logger.debug(f"Extracted price_range: {entities['price_range']}")
                        break
                    except ValueError:
                        logger.debug(f"Invalid price format: {match.group(0)}")

            # --- Weights ---
            weight_pattern = r'(?:بالضبط|حوالي|أكثر\s+من|فوق)?\s*(\d*\.?\d*)\s*(كجم|جم|كيلو|جرام)\b' if lang == 'ar' else r'(?:exactly|around|more\s+than|over)?\s*(\d*\.?\d*)\s*(kg|g|kilogram|gram)\b'
            matches = re.findall(weight_pattern, text_lower, re.IGNORECASE)
            for num, unit in matches:
                try:
                    num = float(num) if num else 0.0
                    logger.debug(f"Weight match: num={num}, unit={unit}, text={text_lower}")
                    if num <= 0.0:
                        logger.debug(f"Skipping invalid weight: {num}{unit}")
                        continue
                    entities["weights"].append((num, unit))
                except ValueError:
                    logger.debug(f"Invalid weight format: {num}{unit}")

            if lang == 'ar':
                if "خفيف" in text_lower and not entities["weights"]:
                    entities["weights"].append("light")
                elif "متوسط" in text_lower and not entities["weights"]:
                    entities["weights"].append("medium")
                elif "تقيل" in text_lower and not entities["weights"]:
                    entities["weights"].append("heavy")
            else:
                if "light" in text_lower and not entities["weights"]:
                    entities["weights"].append("light")
                elif "medium" in text_lower and not entities["weights"]:
                    entities["weights"].append("medium")
                elif "heavy" in text_lower and not entities["weights"]:
                    entities["weights"].append("heavy")
            logger.debug(f"Extracted weights: {entities['weights']}")

            # --- Ratings ---
            rating_patterns = {
                'en': [
                    (r'([0-9]+(?:\.[0-9]+)?)\s*stars?', lambda x: min(5.0, max(0.0, float(x)))),
                    (r'high\s+rated', lambda: 4.0),
                    (r'excellent', lambda: 5.0),
                    (r'very\s+good', lambda: 4.0),
                    (r'good', lambda: 3.0),
                    (r'average', lambda: 2.5),
                    (r'bad', lambda: 1.0),
                ],
                'ar': [
                    (r'([0-9٠١٢٣٤٥٦٧٨٩]+(?:\.[0-9٠١٢٣٤٥٦٧٨٩]+)?)\s*نجوم?', lambda x: min(5.0, max(0.0, float(normalize_arabic(x))))),
                    (r'تقييم\s+عال', lambda: 4.0),
                    (r'ممتاز', lambda: 5.0),
                    (r'جيد\s+جدا', lambda: 4.0),
                    (r'جيد', lambda: 3.0),
                    (r'متوسط', lambda: 2.5),
                    (r'سيء', lambda: 1.0),
                    (r'نجوم\s+أكثر\s+من\s+([0-9٠١٢٣٤٥٦٧٨٩]+(?:\.[0-9٠١٢٣٤٥٦٧٨٩]+)?)', lambda x: min(5.0, max(0.0, float(normalize_arabic(x)))))
                ]
            }

            for pattern, handler in rating_patterns.get(lang, []):
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        if match.lastindex == 2:
                            entities["rating"] = [float(normalize_arabic(match.group(1))), float(normalize_arabic(match.group(2)))]
                        else:
                            entities["rating"] = handler(match.group(1) if match.lastindex else None)
                        logger.debug(f"Extracted rating: {entities['rating']}")
                        break
                    except Exception as e:
                        logger.debug(f"Rating pattern error: {str(e)}")

            # --- Categories ---
            categories = []
            for cat_id in self.products_collection.distinct("category"):
                cat = self.categories_collection.find_one({"_id": cat_id}, {"name": 1})
                if cat and "name" in cat:
                    categories.append(normalize_arabic(cat["name"].lower()) if lang == 'ar' else cat["name"].lower())
            categories = [c.strip() for c in categories if c]

            # --- Subcategories ---
            subcategory_map = {}
            try:
                for sub in self.subcategories_collection.aggregate([
                    {"$lookup": {
                        "from": "categories",
                        "localField": "category",
                        "foreignField": "_id",
                        "as": "category_doc"
                    }},
                    {"$unwind": {
                        "path": "$category_doc",
                        "preserveNullAndEmptyArrays": True
                    }},
                    {"$project": {
                        "name": 1,
                        "category_name": {"$ifNull": ["$category_doc.name", ""]}
                    }}
                ]):
                    if "name" in sub and isinstance(sub["name"], str):
                        subcategory_map[normalize_arabic(sub["name"].lower()) if lang == 'ar' else sub["name"].lower()] = normalize_arabic(sub.get("category_name", "").lower()) if lang == 'ar' else sub.get("category_name", "").lower()
            except Exception as e:
                logger.error(f"Failed to build subcategory_map: {str(e)}")

            location_keywords = ["القاهرة", "الإسكندرية", "الجيزة", "الأقصر", "أسوان"]
            artisans = [normalize_arabic(a["name"].lower()) for a in self.users_collection.find({"role": "artisan"}, {"name": 1}) if "name" in a]
            colors = [normalize_arabic(c) for c in safe_distinct_str("colors")]
            sizes = [normalize_arabic(s) for s in safe_distinct_str("size")]
            weights = safe_distinct("weight")

            # --- Expanded Keyword Map ---
            keyword_map = {
                # Category: Accessories
                "accessories": {"category": "Accessories"},
                "اكسسوارات": {"category": "Accessories"},
                "مستلزمات": {"category": "Accessories"},
                "حاجات اكسسوار": {"category": "Accessories"},
                # Subcategory: Craft Item
                "craft item": {"subcategory": "Craft Item", "category": "Accessories"},
                "عنصر حرفي": {"subcategory": "Craft Item", "category": "Accessories"},
                "منتج يدوي": {"subcategory": "Craft Item", "category": "Accessories"},
                # Category: Ceramics & Pottery
                "ceramics & pottery": {"category": "Ceramics & Pottery"},
                "pottery": {"category": "Ceramics & Pottery"},
                "سيراميك وفخار": {"category": "Ceramics & Pottery"},
                "فخار": {"category": "Ceramics & Pottery"},
                "جرة": {"category": "Ceramics & Pottery"},
                "طواجن": {"category": "Ceramics & Pottery"},
                # Subcategory: Drinkware
                "drinkware": {"subcategory": "Drinkware", "category": "Ceramics & Pottery"},
                "أواني شرب": {"subcategory": "Drinkware", "category": "Ceramics & Pottery"},
                "كوب": {"subcategory": "Drinkware", "category": "Ceramics & Pottery"},
                "كوباية": {"subcategory": "Drinkware", "category": "Ceramics & Pottery"},
                # Subcategory: Tableware
                "tableware": {"subcategory": "Tableware", "category": "Ceramics & Pottery"},
                "أواني مائدة": {"subcategory": "Tableware", "category": "Ceramics & Pottery"},
                "صحون": {"subcategory": "Tableware", "category": "Ceramics & Pottery"},
                # Subcategory: Cooking
                "cooking": {"subcategory": "Cooking", "category": "Ceramics & Pottery"},
                "طهي": {"subcategory": "Cooking", "category": "Ceramics & Pottery"},
                "أدوات طبخ": {"subcategory": "Cooking", "category": "Ceramics & Pottery"},
                # Subcategory: Home Decor
                "home decor": {"subcategory": "Home Decor", "category": "Ceramics & Pottery"},
                "ديكور منزل": {"subcategory": "Home Decor", "category": "Ceramics & Pottery"},
                "زينة بيت": {"subcategory": "Home Decor", "category": "Ceramics & Pottery"},
                # Category: Glass
                "glass": {"category": "Glass"},
                "زجاج": {"category": "Glass"},
                "قزاز": {"category": "Glass"},
                # Category: Leather
                "leather": {"category": "Leather"},
                "جلد": {"category": "Leather"},
                "جلد طبيعي": {"category": "Leather"},
                # Category: Marble
                "marble": {"category": "Marble"},
                "رخام": {"category": "Marble"},
                # Category: Wood
                "wood": {"category": "Wood"},
                "خشب": {"category": "Wood"},
                "عفش خشب": {"category": "Wood"},
                # Colors
                "yellow": {"color": "Yellow"},
                "أصفر": {"color": "Yellow"},
                "صفرا": {"color": "Yellow"},
                "blue": {"color": "Blue"},
                "أزرق": {"color": "Blue"},
                "زرقا": {"color": "Blue"},
                "red": {"color": "Red"},
                "أحمر": {"color": "Red"},
                "حمرا": {"color": "Red"}
            }

            words = [normalize_arabic(w) for w in text_lower.split() if len(w) >= 2]

            for word in words:
                if word in keyword_map:
                    if "category" in keyword_map[word]:
                        entities["categories"].append(keyword_map[word]["category"].capitalize())
                    if "subcategory" in keyword_map[word]:
                        entities["subcategories"].append((
                            keyword_map[word]["subcategory"].capitalize(),
                            keyword_map[word]["category"].capitalize()
                        ))
                    if "color" in keyword_map[word]:
                        entities["colors"].append(keyword_map[word]["color"].capitalize())
                for loc in location_keywords:
                    if fuzz.ratio(word, normalize_arabic(loc)) >= 80:
                        entities["locations"].append(loc.capitalize())

            for word in words:
                for category in categories:
                    if fuzz.ratio(word, category) >= 70:
                        entities["categories"].append(category.capitalize())
                for sub_name in subcategory_map:
                    if fuzz.ratio(word, sub_name) >= 70:
                        cat_name = subcategory_map.get(sub_name, "")
                        entities["subcategories"].append((sub_name.capitalize(), cat_name.capitalize()))
                for artisan in artisans:
                    if fuzz.ratio(word, artisan) >= 80:
                        entities["artisans"].append(artisan.capitalize())
                for color in colors:
                    if fuzz.ratio(word, color) >= 85:
                        entities["colors"].append(color.capitalize())
                for size in sizes:
                    if fuzz.ratio(word, normalize_arabic(str(size).lower())) >= 80:
                        entities["size"].append(str(size).capitalize())
                for weight in weights:
                    if fuzz.ratio(word, normalize_arabic(str(weight).lower())) >= 80:
                        entities["weights"].append(str(weight))

            # --- De-duplication and Cleanup ---
            for key, val in entities.items():
                if isinstance(val, list):
                    entities[key] = list(set(val))
                if key == "subcategories":
                    entities["subcategories"] = [
                        (item[0], item[1]) for item in entities["subcategories"]
                        if isinstance(item, (list, tuple)) and len(item) == 2
                    ]

            logger.debug(f"Final extracted entities: {entities}")

        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            logger.debug(f"Returning partial entities: {entities}")

        return entities

    def _is_partial_filter(self, entities: Dict) -> bool:
        non_empty = [k for k, v in entities.items() if v]
        weak_keys = {"price_range", "rating", "colors", "materials", "size", "weights", "locations"}
        strong_keys = {"categories", "subcategories", "product_titles", "artisans"}

        weak = sum(1 for k in non_empty if k in weak_keys)
        strong = sum(1 for k in non_empty if k in strong_keys)

        return strong == 0 and weak < 1

    def _extract_price_range(self, text: str, lang: str) -> Optional[Tuple[float, float]]:
        """Extract price range from text in both English and Arabic."""
        logger.debug(f"Extracting price range from text: '{text}' (lang: {lang})")
        try:
            # Normalize Arabic numerals to ASCII digits
            arabic_to_ascii = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            normalized_text = text.translate(arabic_to_ascii).lower()

            en_patterns = [
                (r'under\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x: (0.0, float(x))),
                (r'less\s+than\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x: (0.0, float(x))),
                (r'below\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x: (0.0, float(x))),
                (r'over\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x: (float(x), float('inf'))),
                (r'more\s+than\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x: (float(x), float('inf'))),
                (r'above\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x: (float(x), float('inf'))),
                (r'between\s+(\d*\.?\d+)\s*(?:and|to)\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x, y: (float(x), float(y))),
                (r'from\s+(\d*\.?\d+)\s*(?:to|until)\s+(\d*\.?\d+)\s*(?:egp|pounds?|le)?(?!\s*(kg|g|kilogram|gram))', 
                lambda x, y: (float(x), float(y))),
                (r'cheap', lambda: (0.0, 200.0)),
                (r'expensive', lambda: (500.0, float('inf')))
            ]

            ar_patterns = [
                (r'أقل\s+من\s+(\d*\.?\d+)\s*(?:جنيه|ج)?(?!\s*(كجم|جم|كيلوجرام|جرام))', 
                lambda x: (0.0, float(x))),
                (r'تحت\s+(\d*\.?\d+)\s*(?:جنيه|ج)?(?!\s*(كجم|جم|كيلوجرام|جرام))', 
                lambda x: (0.0, float(x))),  # Fixed handler
                (r'أكثر\s+من\s+(\d*\.?\d+)\s*(?:جنيه|ج)?(?!\s*(كجم|جم|كيلوجرام|جرام))', 
                lambda x: (float(x), float('inf'))),
                (r'فوق\s+(\d*\.?\d+)\s*(?:جنيه|ج)?(?!\s*(كجم|جم|كيلوجرام|جرام))', 
                lambda x: (float(x), float('inf'))),
                (r'بين\s+(\d*\.?\d+)\s*(?:و|إلى)\s+(\d*\.?\d+)\s*(?:جنيه|ج)?(?!\s*(كجم|جم|كيلوجرام|جرام))', 
                lambda x, y: (float(x), float(y))),
                (r'من\s+(\d*\.?\d+)\s*(?:إلى|حتى)\s+(\d*\.?\d+)\s*(?:جنيه|ج)?(?!\s*(كجم|جم|كيلوجرام|جرام))', 
                lambda x, y: (float(x), float(y))),
                (r'رخيص', lambda: (0.0, 200.0)),
                (r'غالي', lambda: (500.0, float('inf')))
            ]

            # Prioritize language-specific patterns
            patterns_to_try = ar_patterns if lang == "ar" else en_patterns

            for pattern, handler in patterns_to_try:
                match = re.search(pattern, normalized_text, re.IGNORECASE)
                if match:
                    try:
                        logger.debug(f"Price match: groups={match.groups()}")
                        groups = [g for g in match.groups() if g is not None]  # Filter out None
                        if not groups:  # Keyword-based (e.g., "cheap")
                            min_price, max_price = handler()
                        elif len(groups) == 1:
                            min_price, max_price = handler(groups[0])
                        else:
                            min_price, max_price = handler(groups[0], groups[1])

                        # Validate price range
                        if not (isinstance(min_price, (int, float)) and isinstance(max_price, (int, float))):
                            logger.warning(f"Invalid price types: min={min_price}, max={max_price}")
                            continue
                        if min_price < 0 or (max_price != float('inf') and max_price < 0):
                            logger.warning(f"Negative price range: {min_price}, {max_price}")
                            continue
                        if max_price != float('inf') and min_price > max_price:
                            logger.warning(f"Invalid price range: min={min_price} > max={max_price}")
                            min_price, max_price = max_price, min_price

                        result = (float(min_price), float(max_price))
                        logger.debug(f"Extracted price range: {result}")
                        return result
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error processing price pattern {pattern}: {str(e)}")
                        continue

            logger.debug("No price range matched")
            return None

        except Exception as e:
            logger.warning(f"Error extracting price range: {str(e)}")
            return None

    def _classify_intent(self, text: str, lang: str) -> Dict:
        """Classify intent with keyword-based rules and entities. Returns: {'intent': str, 'confidence': float, 'entities': Dict}"""
        logger.debug(f"Classifying intent for text: '{text}' (lang: {lang})")
        try:
            if not lang:
                lang = detect(text)

            def normalize_arabic(text):
                arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
                text = re.sub(arabic_diacritics, '', text)
                arabic_numerals = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                return text.translate(arabic_numerals)

            text_lower = normalize_arabic(text.lower()) if lang == 'ar' else text.lower()

            entities = self._extract_entities(text, lang)
            if not isinstance(entities, dict):
                logger.error(f"Invalid entities format: {entities}")
                return {"intent": "clarification", "confidence": 0.3, "entities": {}}

            # Initialize flags
            has_category = bool(entities.get("categories", []) or entities.get("subcategories", []))
            has_price = bool(entities.get("price_range") and isinstance(entities["price_range"], (tuple, list)) and len(entities["price_range"]) == 2)
            has_rating = entities.get("rating") is not None
            has_location = bool(entities.get("locations", []))
            has_product_title = bool(entities.get("product_titles", []))
            has_artisan = bool(entities.get("artisans", []))
            has_color = bool(entities.get("colors", []))
            has_size = bool(entities.get("size", []))
            has_weight = bool(entities.get("weights", []))

            # Define keywords for intents
            greeting_keywords = {
                "en": ["hello", "hi", "hey", "greetings"],
                "ar": ["مرحبا", "مرحبًا", "أهلا", "اهلا", "سلام", "ازيك"]
            }
            recommendation_keywords = {
                "en": ["recommend", "suggest", "what should i", "best"],
                "ar": ["يوصي", "اقترح", "أفضل", "نصحيني", "إيه الأحسن"]
            }
            help_keywords = {
                "en": ["help", "how to", "support", "assist"],
                "ar": ["مساعدة", "كيف", "دعم", "ساعدني", "إزاي"]
            }
            feedback_keywords = {
                "en": ["feedback", "review", "rate", "opinion"],
                "ar": ["تعليق", "مراجعة", "تقييم", "رأي", "قولي رأيك"]
            }
            product_query_keywords = {
                "en": ["show me", "i want", "find", "get me", "looking for", "need", "search for"],
                "ar": ["عايز", "أريد", "أظهر", "ابحث", "حابب", "بدور على", "محتاج", "لقّيني", "جيبلي"]
            }
            filter_keywords = {
                "en": ["under", "below", "less than", "more than", "filter", "show me", "in", "from", "to"],
                "ar": ["تحت", "أقل من", "أكثر من", "فلتر", "أظهر", "في", "من", "إلى", "بأقل", "بأكتر"]
            }
            price_query_keywords = {
                "en": ["price", "cost", "expensive", "cheap", "budget", "affordable"],
                "ar": ["سعر", "تكلفة", "غالي", "رخيص", "ميزانية", "بكام", "السعر إيه"]
            }

            # Intent classification logic
            if any(kw in text_lower for kw in greeting_keywords.get(lang, [])):
                logger.debug("Greeting keywords detected")
                return {"intent": "greeting", "confidence": 0.9, "entities": entities}
            elif any(kw in text_lower for kw in recommendation_keywords.get(lang, [])):
                logger.debug("Recommendation keywords detected")
                return {"intent": "recommendation", "confidence": 0.85, "entities": entities}
            elif any(kw in text_lower for kw in help_keywords.get(lang, [])):
                logger.debug("Help keywords detected")
                return {"intent": "help", "confidence": 0.85, "entities": entities}
            elif any(kw in text_lower for kw in feedback_keywords.get(lang, [])):
                logger.debug("Feedback keywords detected")
                return {"intent": "feedback", "confidence": 0.85, "entities": entities}
            elif has_category and not has_product_title and any(kw in text_lower for kw in product_query_keywords.get(lang, [])):
                logger.debug("Category without product titles, setting intent to category_query")
                return {"intent": "category_query", "confidence": 0.9, "entities": entities}
            elif (has_price or has_color or has_size or has_weight or has_rating or has_location or has_artisan) and any(kw in text_lower for kw in filter_keywords.get(lang, [])):
                logger.debug("Filter attributes detected, setting intent to filter")
                return {"intent": "filter", "confidence": 0.9, "entities": entities}
            elif has_product_title and any(kw in text_lower for kw in product_query_keywords.get(lang, [])):
                logger.debug("Product titles detected, setting intent to product_query")
                return {"intent": "product_query", "confidence": 0.9, "entities": entities}
            elif any(kw in text_lower for kw in price_query_keywords.get(lang, [])):
                logger.debug("Price query keywords detected")
                return {"intent": "price_query", "confidence": 0.8, "entities": entities}
            elif has_color or has_category:
                logger.debug("Vague color or category query, setting intent to filter")
                return {"intent": "filter", "confidence": 0.7, "entities": entities}
            else:
                logger.debug("No clear intent matched, defaulting to clarification")
                return {"intent": "clarification", "confidence": 0.3, "entities": entities}

        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            return {"intent": "clarification", "confidence": 0.3, "entities": {}}

    def _check_conflicting_entities(self, entities: Dict, lang: str) -> Optional[Dict]:
        """
        Check for conflicting or invalid entities in the user's input.
        Returns a clarification response if conflicts are found, None otherwise.
        """
        try:
            price_terms = {
                'en': {
                    'cheap': ['cheap', 'low', 'affordable', 'budget', 'inexpensive'],
                    'expensive': ['expensive', 'high', 'luxury', 'premium', 'costly']
                },
                'ar': {
                    'cheap': ['رخيص', 'رخيصة', 'رخيصين', 'رخيصه', 'رخيصين', 'مش غالية'],
                    'expensive': ['غالي', 'غالية', 'غالين', 'غاليه', 'غالين', 'مكلف', 'مكلفة', 'مكلفين']
                }
            }
            
            price_range = entities.get("price_range")
            if (
                price_range and 
                isinstance(price_range, (tuple, list)) and 
                len(price_range) == 2 and 
                isinstance(price_range[0], (int, float)) and 
                isinstance(price_range[1], (int, float))
            ):
                min_price, max_price = price_range
                if min_price > max_price:
                    return {
                        "status": "clarify",
                        "response": self.responses[lang]["invalid_price_range"][0],
                        "suggestions": [
                            self.responses[lang]["price_range_example"][0],
                            self.responses[lang]["price_range_example"][1]
                        ],
                        "confidence": 0.0,
                        "intent": "clarify",
                        "lang": lang,
                        "entities": entities,
                        "needs_clarification": True
                    }
            
            original_message = self.context.get("original_message", "").lower()
            has_cheap = any(term in original_message for term in price_terms[lang]['cheap'])
            has_expensive = any(term in original_message for term in price_terms[lang]['expensive'])
            
            if has_cheap and has_expensive:
                return {
                    "status": "clarify",
                    "response": self.responses[lang]["conflicting_price"][0],
                    "suggestions": [
                        self.responses[lang]["price_clarification"][0],
                        self.responses[lang]["price_clarification"][1]
                    ],
                    "confidence": 0.0,
                    "intent": "clarify",
                    "lang": lang,
                    "entities": entities,
                    "needs_clarification": True
                }
            
            rating = entities.get("rating")
            if rating is not None:
                if rating < 0 or rating > 5:
                    return {
                        "status": "clarify",
                        "response": self.responses[lang]["invalid_rating"][0],
                        "suggestions": [
                            self.responses[lang]["rating_example"][0],
                            self.responses[lang]["rating_example"][1]
                        ],
                        "confidence": 0.0,
                        "intent": "clarify",
                        "lang": lang,
                        "entities": entities,
                        "needs_clarification": True
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking conflicting entities: {str(e)}")
            return None

    def _preprocess_text(self, message: str, lang: str) -> Tuple[str, bool]:
        """
        Preprocess the input message and return the cleaned version and is_mixed flag.
        """
        try:
            # Detect English and Arabic characters more precisely
            ascii_letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            arabic_chars = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")
            has_english = any(c in ascii_letters for c in message)
            has_arabic = any(c in arabic_chars for c in message)
            is_mixed = has_english and has_arabic

            # Skip preprocessing for pure English inputs
            if lang == "en" and not is_mixed:
                return message.strip(), False

            # Apply preprocessing for Arabic or mixed-language inputs
            if lang == "ar" or is_mixed:
                message = self.split_mixed_script_tokens(message)
                message = self.replace_transliterated_words(message)
                message = self.normalize_arabic(message)
                message = self.expand_dialect_variants(message)

            return message.strip(), is_mixed

        except Exception as e:
            logger.error(f"Error in message preprocessing: {str(e)}")
            raise LanguageError(f"Failed to preprocess message: {str(e)}")
    
    def process_message(self, message: str, lang: str, intent: str = None, entities: Dict = None) -> Dict:
        try:
            if not message or not message.strip():
                raise ValidationError("Empty message")

            is_mixed = False
            original_message = message

            try:
                arabic_chars = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")
                ascii_letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                if lang == "en" and any(c in arabic_chars for c in message):
                    is_mixed = True
                elif lang == "ar" and any(c in ascii_letters for c in message):
                    is_mixed = True

                if is_mixed or lang == "ar":
                    message = ChatbotService.replace_transliterated_words(self, message)
                    message = ChatbotService.split_mixed_script_tokens(self, message)
                message = self._preprocess_text(message, lang)[0]

                if lang == "en" and any(c in arabic_chars for c in message):
                    is_mixed = True
                elif lang == "ar" and any(c in ascii_letters for c in message):
                    is_mixed = True

                self.context["last_message"] = message
                self.context["original_message"] = original_message
                logger.debug(f"Original message: {original_message}")
                logger.debug(f"Preprocessed message: {message} (mixed: {is_mixed})")
            except Exception as e:
                logger.error(f"Error in message preprocessing: {str(e)}")
                raise LanguageError(f"Failed to preprocess message: {str(e)}")

            # Use provided intent and entities
            if not intent or not entities:
                try:
                    entities = self._extract_entities(message, lang)
                    logger.debug(f"Extracted entities: {entities}")
                    result = self._classify_intent(message, lang)
                    intent = result["intent"]
                    confidence = result["confidence"]
                    entities = result["entities"]
                    logger.debug(f"Classified intent: {intent} with confidence {confidence}")
                except Exception as e:
                    logger.error(f"Error in intent classification: {str(e)}")
                    raise IntentClassificationError(f"Failed to classify intent: {str(e)}")
            else:
                confidence = 0.9  # Assume high confidence for passed intent
                logger.debug(f"Using provided intent: {intent}, entities: {entities}")

            conflict_response = self._check_conflicting_entities(entities, lang)
            if conflict_response:
                return conflict_response

            # Relaxed clarification: proceed if confidence is reasonable or entities exist
            if intent not in ["greeting", "help"] and confidence < 0.5 and not any([
                entities.get("categories"), entities.get("price_range"),
                entities.get("product_titles"), entities.get("rating"), entities.get("colors")
            ]):
                return self._format_clarification_response(lang, intent, confidence, entities)

            response_lang = lang
            response = {}

            # Handle search-related intents
            if intent in {"product_query", "category_query", "filter"}:
                try:
                    if intent == "product_query":
                        response = self._handle_product_query(entities, response_lang)
                    elif intent == "category_query":
                        response = self._handle_category_query(entities, response_lang)
                    elif intent == "filter":
                        response = self._handle_filter(entities, response_lang)

                    if response.get("products"):
                        response["products"] = response["products"][:5]
                        response["status"] = "success"
                    else:
                        response["status"] = "no_results"
                except Exception as e:
                    logger.error(f"Error handling intent {intent}: {str(e)}")
                    response = {
                        "response": self.responses[lang]["no_results"][0],
                        "status": "no_results",
                        "products": [],
                        "suggestions": self._format_suggestions({"intent": intent, "entities": entities, "lang": lang})
                    }
            else:
                try:
                    response = self._handle_intent(intent, entities, response_lang)
                except Exception as e:
                    logger.error(f"Error handling intent: {str(e)}")
                    if intent == "price_query":
                        raise PriceQueryError(f"Failed to handle price query: {str(e)}")
                    elif intent == "product_query":
                        raise ProductQueryError(f"Failed to handle product query: {str(e)}")
                    else:
                        raise ChatbotError(f"Failed to handle intent {intent}: {str(e)}")

            response.update({
                "confidence": confidence,
                "intent": intent,
                "lang": lang,
                "entities": entities,
                "is_mixed_language": is_mixed
            })

            if app.debug:
                response["debug"] = {
                    "entities": entities,
                    "confidence": confidence,
                    "intent": intent
                }
            logger.debug(f"Response: {response}")
            return response

        except ChatbotError as e:
            return self._handle_error(e, lang)
        except Exception as e:
            logger.error(f"Unexpected error in process_message: {str(e)}\n{traceback.format_exc()}")
            return self._handle_error(e, lang)

    def _handle_intent(self, intent: str, entities: Dict, lang: str) -> Dict:
        """Handle general (non-product) intents with appropriate responses"""
        try:
            if intent == "greeting":
                return self._handle_greeting(lang)
            elif intent == "recommendation":
                return self._handle_recommendation(lang)
            elif intent == "help":
                return self._handle_help(lang)
            elif intent == "feedback":
                return self._handle_feedback(lang)
            elif intent == "price_query":
                return self._handle_price_query(entities, lang)
            else:
                return {
                    "status": "clarify",
                    "response": self.responses[lang]["clarify"][0],
                    "suggestions": self._format_suggestions({"intent": "clarification", "entities": entities, "lang": lang}),
                    "intent": "clarification",
                    "confidence": 0.0,
                    "entities": entities
                }
        except Exception as e:
            logger.error(f"Error handling intent {intent}: {str(e)}")
            return self._handle_error(e, lang)

    def _format_product(self, product: Dict) -> Optional[Dict]:
        """Ensure a product document is JSON-serializable and all ObjectIds are converted to strings."""
        try:
            if not product:
                return None

            # Convert top-level _id
            if '_id' in product:
                product['_id'] = str(product['_id'])

            # Convert ObjectId references
            if 'category' in product and isinstance(product['category'], ObjectId):
                product['category'] = str(product['category'])
            if 'subcategories' in product and isinstance(product['subcategories'], list):
                product['subcategories'] = [str(sc) if isinstance(sc, ObjectId) else sc for sc in product['subcategories']]
            if 'artisan' in product and isinstance(product['artisan'], ObjectId):
                product['artisan'] = str(product['artisan'])

            # Convert any datetime to ISO format
            for key, value in product.items():
                if isinstance(value, datetime):
                    product[key] = value.isoformat()
                elif isinstance(value, (np.ndarray, torch.Tensor)):
                    product[key] = value.tolist()

            return product

        except Exception as e:
            logger.error(f"Error formatting product: {str(e)}")
            return None


    def _format_clarification_response(self, lang: str, intent: str, confidence: float, entities: Dict) -> Dict:
        """Format a clarification response based on the context."""
        try:
            base_message = self.responses[lang]["clarify"][0]
            
            if intent == "product_query":
                if not (entities.get("categories") or entities.get("subcategories")):
                    base_message += f" {self.responses[lang]['clarify_category'][0]}"
                elif not entities.get("price_range"):
                    base_message += f" {self.responses[lang]['clarify_price'][0]}"
            
            suggestions = self._get_sample_queries(lang)
            
            return {
                "status": "clarify",
                "response": base_message,
                "suggestions": suggestions,
                "confidence": confidence,
                "intent": intent,
                "lang": lang,
                "entities": entities,
                "needs_clarification": True
            }
            
        except Exception as e:
            logger.error(f"Error formatting clarification response: {str(e)}")
            return {
                "status": "clarify",
                "response": self.responses[lang]["clarify"][0],
                "suggestions": self._get_sample_queries(lang),
                "confidence": confidence,
                "intent": intent,
                "lang": lang,
                "entities": entities
            }


    def _format_suggestions(self, params: dict) -> list:
        """Return example suggestions the user can try based on intent and language."""
        lang = params.get("lang", "en")
        intent = params.get("intent", "unknown")
        # entities = params.get("entities", {})  # Not used currently, but keep for future

        suggestions_bank = {
            "en": {
                "filter": ["Show me cheap bags", "Products under 100 EGP", "Filter by price"],
                "product_query": ["Handmade bracelet", "Wooden keychain", "Best rated products"],
                "price_query": ["Products between 100 and 200", "Show me items under 300 EGP"],
                "category_query": ["Show me accessories", "What's in home decor?"],
                "greeting": ["What can I ask?", "Browse products", "Start over"],
                "unknown": ["Help", "Show all products", "What's popular now?"]
            },
            "ar": {
                "filter": ["وريني شنط رخيصة", "منتجات اقل من ١٠٠ جنيه", "فلتر بالسعر"],
                "product_query": ["اسوارة يد", "ميدالية خشب", "منتجات تقييمها عالي"],
                "price_query": ["منتجات بين ١٠٠ و ٢٠٠", "حاجه تحت ٣٠٠ جنيه"],
                "category_query": ["وريني قسم الاكسسوارات", "ايه عندكم في الديكور؟"],
                "greeting": ["ابدأ من جديد", "ايه اقدر اسأل؟", "تصفح المنتجات"],
                "unknown": ["مساعدة", "ايه الاشهر عندكم؟", "اعرضلي كل المنتجات"]
            }
        }

        lang_suggestions = suggestions_bank.get(lang, suggestions_bank["en"])
        return lang_suggestions.get(intent, lang_suggestions["unknown"])


    def _handle_error(self, error, lang):
        """Enhanced error handling with proper language support and logging"""
        try:
            error_message = str(error)
            logger.error(f"Error occurred: {error_message}\n{traceback.format_exc()}")

            error_messages = self.responses.get(lang, {}).get("error_messages", {})
            error_type = error.__class__.__name__

            # 🔍 Classify error type if not a custom ChatbotError
            if not isinstance(error, ChatbotError):
                if isinstance(error, (ValueError, TypeError, KeyError, AttributeError, IndexError)):
                    error_type = "ValidationError"
                elif isinstance(error, (ConnectionError, TimeoutError, PyMongoError)):
                    error_type = "DatabaseError"
                elif isinstance(error, (UnicodeError, LookupError)):
                    error_type = "LanguageError"
                else:
                    error_type = "default"

            response = error_messages.get(error_type, error_messages.get("default", "An unexpected error occurred."))

            error_details = None
            if app.debug:
                error_details = {
                    'type': error_type,
                    'message': error_message,
                    'traceback': traceback.format_exc()
                }

            return {
                "intent": "error",
                "response": response,
                "error": error_message,
                "error_details": error_details,
                "entities": {},
                "status": "error"
            }

        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}\n{traceback.format_exc()}")
            return {
                "intent": "error",
                "response": "An unexpected error occurred. Please try again.",
                "error": str(e),
                "entities": {},
                "status": "error"
            }
    
    # Enhanced Arabic dialect variants dictionary
    ARABIC_DIALECT_VARIANTS = {
        "want": ["عايز", "3ayez", "عاوز", "ابغى", "اريد", "عوز", "عايزة", "عاوزة", "ابغا", "ابغي"],
        "need": ["محتاج", "7taj", "محتاجة", "محتاچ", "محتاچة"],
        "like": ["بحب", "b7eb", "بحبها", "بحبه", "بحبو", "بحبهم"],
        "buy": ["اشترى", "اشتري", "اشتريت", "اشترى", "اشتريت", "اشتريت", "اشتريت", "اشتريت"],
        "see": ["شوف", "shoof", "شايف", "شايفة", "شايفين", "شايفين"],
        "cheap": ["رخيص", "rakhees", "رخيصة", "مش غالية", "رخيصه", "رخيصين", "رخيصين"],
        "expensive": ["غالي", "غالية", "مكلف", "غاليه", "غالين", "غالين", "مكلفة", "مكلفين"],
        "price": ["سعر", "السعر", "التكلفة", "التكلفه", "السعر", "السعر", "السعر"],
        "discount": ["خصم", "الخصم", "تخفيض", "التخفيض", "تخفيض", "التخفيض"],
        "product": ["منتج", "منتجات", "items", "حاجة", "حاجات", "شيء", "اشياء", "شي", "اشياء"],
        "quality": ["جودة", "الجودة", "نوعية", "النوعية", "نوعية", "النوعية"],
        "new": ["جديد", "جديدة", "جديدين", "جديدين", "جديده", "جديده"],
        "old": ["قديم", "قديمة", "قديمين", "قديمين", "قديمه", "قديمه"],
        "beautiful": ["جميل", "جميلة", "جميلين", "جميلين", "جميله", "جميله", "حلو", "حلوة", "حلوين", "حلوين"],
        "good": ["كويز", "كويس", "كويسة", "كويسين", "كويسين", "كويسه", "كويسه"],
        "bad": ["وحش", "وحشة", "وحشين", "وحشين", "وحشه", "وحشه", "سيء", "سيئة", "سيئين", "سيئين"],
        "shop": ["متجر", "المتجر", "محل", "المحل", "دكان", "الدكان"],
        "store": ["متجر", "المتجر", "محل", "المحل", "دكان", "الدكان"],
        "seller": ["بائع", "البائع", "بائعة", "البائعة", "بائعين", "البائعين"],
        "artisan": ["حرفي", "الحرفي", "حرفية", "الحرفية", "حرفيين", "الحرفيين"]
    }

    # Regular expressions for Arabic text normalization
    ARABIC_DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
    TATWEEL = "\u0640"
    ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    ARABIC_PUNCTUATION = re.compile(r'[\u060C\u060D\u060E\u060F\u061B\u061E\u061F\u066A-\u066C\u06D4\u06F7-\u06F9]')
    ARABIC_LIGATURES = re.compile(r'[\uFDF2\uFDFA\uFDFB]')
    
    def normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text by:
        1. Removing diacritics
        2. Removing Tatweel
        3. Converting Arabic-Indic numbers to standard digits
        4. Normalizing punctuation
        5. Removing ligatures
        6. Trimming whitespace
        
        Args:
            text (str): Input Arabic text
            
        Returns:
            str: Normalized Arabic text
        """
        try:
            if not text:
                return ""
                
            # Convert to string if not already
            text = str(text)
            
            # Remove diacritics
            text = self.ARABIC_DIACRITICS.sub('', text)
            
            # Remove Tatweel
            text = text.replace(self.TATWEEL, '')
            
            # Convert Arabic-Indic numbers to standard digits
            text = text.translate(self.ARABIC_INDIC_DIGITS)
            
            # Normalize punctuation
            text = self.ARABIC_PUNCTUATION.sub(' ', text)
            
            # Remove ligatures
            text = self.ARABIC_LIGATURES.sub('', text)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in Arabic normalization: {str(e)}")
            return text  # Return original text on error

    def expand_dialect_variants(self, text: str) -> str:
        """
        Expand dialect variants by appending their canonical forms only when not already present.
        """
        try:
            if not text:
                return ""

            words = text.split()
            expanded_words = list(words)  # Copy to allow safe appending

            for word in words:
                if len(word) < 3:
                    continue

                for canonical, variants in self.ARABIC_DIALECT_VARIANTS.items():
                    if word in variants and canonical not in expanded_words:
                        expanded_words.append(canonical)
                        break  # Avoid appending multiple times

            return " ".join(expanded_words)

        except Exception as e:
            logger.error(f"Error in dialect variant expansion: {str(e)}")
            return text  # Fallback to original

    def _get_sample_queries(self, lang: str) -> List[str]:
        """Get sample queries in the user's language for clarification."""
        sample_queries = {
            'en': [
                "Show me some handmade products",
                "What products do you have in the home decor category?",
                "I'm looking for something under 500 EGP",
                "Do you have any products from Cairo?",
                "Show me products with 4+ star ratings"
            ],
            'ar': [
                "عايز اشوف منتجات يدوية",
                "عندك ايه في قسم الديكور؟",
                "عايز حاجة تحت ٥٠٠ جنيه",
                "عندك منتجات من القاهرة؟",
                "عايز منتجات تقييمها ٤ نجوم واكتر"
            ]
        }
        return sample_queries.get(lang, sample_queries['en'])

    def _initialize_product_index(self):
        """Initialize the product search index"""
        try:
            # Get all products from MongoDB
            products = list(self.products_collection.find())
            if products:
                # Convert ObjectId to string for JSON serialization
                for product in products:
                    product['_id'] = str(product['_id'])
                # Initialize FAISS index
                self.embedding_service.initialize_product_index(products)
                logger.info(f"Initialized product index with {len(products)} products")
            else:
                logger.warning("No products found in database")
        except Exception as e:
            logger.error(f"Error initializing product index: {str(e)}")
            raise DatabaseError("Failed to initialize product search", "en")

    def is_mongo_connected(self):
        try:
            # Ping the database to check connection
            self.client.admin.command('ping')
            return True
        except Exception:
            return False

    def _filter_by_entities(self, products: List[Dict], entities: Dict) -> List[Dict]:
        """Re-apply structured filters to FAISS-matched products"""
        filtered = []
        for product in products:
            try:
                price = product.get("priceAfterDiscount", product.get("price", 0))
                if entities.get("price_range"):
                    min_p, max_p = entities["price_range"]
                    if not (min_p <= price <= (max_p if max_p != float("inf") else 1e6)):
                        continue

                if entities.get("rating") is not None:
                    if product.get("ratingsAverage", 0) < entities["rating"]:
                        continue

                if entities.get("colors") and not set(product.get("colors", [])) & set(entities["colors"]):
                    continue

                if entities.get("size") and product.get("size") not in entities["size"]:
                    continue

                if entities.get("materials") and not set(product.get("materials", [])) & set(entities["materials"]):
                    continue

                if entities.get("weights"):
                    normalized_weight = self.embedding_service.normalize_weight(product.get("weight", ""))
                    logger.debug(f"Weight filter: product_weight={normalized_weight}, expected={entities['weights']}")
                    if normalized_weight not in entities["weights"]:
                        continue

                if entities.get("categories"):
                    category = product.get("category")
                    product_category = ""
                    if isinstance(category, dict):
                        product_category = category.get("name", "").lower()
                    elif isinstance(category, str):
                        product_category = category.lower()
                    else:
                        continue  # skip if ObjectId

                    keywords = [cat.lower() for cat in entities["categories"]]
                    if not any(kw in product_category for kw in keywords):
                        continue

                if entities.get("subcategories"):
                    matched_sub = False
                    for sub in product.get("subcategories", []):
                        sub_name = (
                            sub.get("name", "").lower() if isinstance(sub, dict)
                            else sub.lower() if isinstance(sub, str)
                            else ""
                        )
                        for expected in entities["subcategories"]:
                            expected_name = expected[0].lower() if isinstance(expected, tuple) else str(expected).lower()
                            if expected_name in sub_name:
                                matched_sub = True
                                break
                        if matched_sub:
                            break
                    if not matched_sub:
                        continue

                if entities.get("artisans"):
                    artisan = product.get("artisan")
                    artisan_name = (
                        artisan.get("name", "").lower() if isinstance(artisan, dict)
                        else artisan.lower() if isinstance(artisan, str)
                        else ""
                    )
                    artisan_keywords = [a.lower() for a in entities["artisans"]]
                    if not any(kw in artisan_name for kw in artisan_keywords):
                        continue

                # Loose fallback if no strong filter applied
                if not any([
                    entities.get("categories"), entities.get("subcategories"),
                    entities.get("materials"), entities.get("colors"),
                    entities.get("size"), entities.get("locations"),
                    entities.get("artisans"), entities.get("price_range"),
                    entities.get("rating")
                ]):
                    query_text = (self.context.get("original_message") or "").lower()
                    product_text = f"{product.get('title', '')} {product.get('description', '')}".lower()
                    keyword_tokens = [word for word in query_text.split() if len(word) >= 3]
                    if not any(kw in product_text for kw in keyword_tokens):
                        continue

                filtered.append(product)
            except Exception as e:
                logger.warning(f"Error re-filtering FAISS result: {str(e)}")
                continue

        logger.debug(f"Filtered {len(filtered)} products from {len(products)}")
        return filtered

    def _safe_faiss_search(self, query_text: str, lang: str, entities: Dict, intent: str) -> Dict:
        """Perform FAISS search with normalization and robust error handling."""
        logger.debug(f"FAISS search for: '{query_text}' | intent: '{intent}' | entities: {entities}")

        response = {
            "status": "no_results",
            "response": self.responses[lang]["no_results"][0],
            "products": [],
            "suggestions": self._format_suggestions(lang, intent=intent, entities=entities)
        }

        try:
            if not query_text.strip() or not self.embedding_service or not self.product_lookup:
                logger.warning("Missing query text or FAISS index.")
                return response

            # Normalize query text
            query_text = self.normalize_text(query_text)

            faiss_results = self.embedding_service.search_products(query_text, k=20)
            logger.debug(f"FAISS returned {len(faiss_results)} results")

            if faiss_results and isinstance(faiss_results[0], dict) and faiss_results[0].get("_faiss_error"):
                logger.warning("FAISS error detected, fallback triggered.")
                response["status"] = "faiss_unavailable"
                response["response"] = self.responses[lang]["faiss_down"][0]
                return response

            full_products = [
                self.product_lookup.get(p["_id"]) for p in faiss_results if p.get("_id") in self.product_lookup
            ]
            full_products = [p for p in full_products if p is not None]
            logger.debug(f"Retrieved {len(full_products)} products from product_lookup")

            filtered = self._filter_by_entities(full_products, entities)
            final_results = filtered[:5] if filtered else full_products[:5]

            formatted_products = [
                self._format_product_response(self._format_product(p), lang)
                for p in final_results
            ]

            if formatted_products:
                response["status"] = "semantic_results"
                response["response"] = self.responses[lang]["product_query"][0]
                response["products"] = formatted_products
                response["source"] = "semantic+filtered" if filtered else "semantic"
            else:
                response["status"] = "no_results"
                response["response"] = self.responses[lang]["no_results"][0]

            return response

        except Exception as e:
            logger.error(f"FAISS search failed: {str(e)}")
            response["status"] = "error"
            response["response"] = self.responses[lang]["error"][0]
            return response
    
    def _matches_entities(self, product: Dict, entities: Dict) -> bool:
        """Check if a product matches the structured entities."""
        try:
            # Price range
            if entities.get("price_range"):
                min_p, max_p = entities["price_range"]
                price = product.get("priceAfterDiscount", product.get("price", 0))
                if not (min_p <= price <= (max_p if max_p != float("inf") else 1e6)):
                    return False

            # Color
            if entities.get("colors"):
                product_colors = [c.lower() for c in product.get("colors", [])]
                if not any(c.lower() in product_colors for c in entities["colors"]):
                    return False

            # Size
            if entities.get("size"):
                size = str(product.get("size", "")).lower()
                if not any(s.lower() == size for s in entities["size"]):
                    return False

            # Weight
            if entities.get("weights"):
                weight = str(product.get("weight", "")).lower()
                if not any(w.lower() == weight for w in entities["weights"]):
                    return False

            # Rating
            if entities.get("rating") is not None:
                if float(product.get("ratingsAverage", 0)) < float(entities["rating"]):
                    return False

            # Location (match in title/description)
            if entities.get("locations"):
                text = f"{product.get('title', '')} {product.get('description', '')}".lower()
                if not any(loc.lower() in text for loc in entities["locations"]):
                    return False

            # Artisan
            if entities.get("artisans"):
                artisan_name = ""
                artisan = product.get("artisan")
                if isinstance(artisan, ObjectId):
                    doc = self.products_collection["users"].find_one(
                        {"_id": artisan}, {"name": 1}
                    )
                    artisan_name = doc.get("name", "").lower() if doc else ""
                elif isinstance(artisan, dict):
                    artisan_name = artisan.get("name", "").lower()
                elif isinstance(artisan, str):
                    artisan_name = artisan.lower()
                else:
                    return False
                if not any(a.lower() in artisan_name for a in entities["artisans"]):
                    return False

            return True

        except Exception as e:
            logger.warning(f"Entity filter error: {str(e)}")
            return False
    
    def _handle_product_query(self, entities: Dict, lang: str) -> Dict:
        """Handle product query intent with precise MongoDB query for product titles."""
        logger.debug(f"Handling product query with entities: {entities}")
        try:
            response = {
                "status": "success",
                "response": "",
                "products": [],
                "intent": "product_query",
                "confidence": 0.9,
                "entities": entities
            }

            # Product Titles
            product_titles = entities.get("product_titles", [])
            if not product_titles:
                logger.debug("No product titles provided, returning clarification")
                response.update({
                    "status": "clarify",
                    "response": self.responses[lang]["clarify"][0],
                    "suggestions": self._format_suggestions({"intent": "product_query", "entities": entities, "lang": lang}),
                    "intent": "clarification",
                    "confidence": 0.5
                })
                return response

            # Build query for exact product titles
            title_regex = '|'.join([re.escape(title) for title in product_titles])
            query = {"title": {"$regex": f"^{title_regex}$", "$options": "i"}}
            logger.debug(f"Product query: {query}")

            # Execute Query
            products = list(self.products_collection.find(query, {
                "title": 1, "description": 1, "price": 1, "priceAfterDiscount": 1,
                "imageCover": 1, "colors": 1, "size": 1, "weight": 1, "ratingsAverage": 1,
                "category": 1, "subcategories": 1, "artisan": 1
            }).limit(10))

            if not products:
                logger.debug("No products found")
                response.update({
                    "status": "no_results",
                    "response": self.responses[lang]["no_results"][0],
                    "products": [],
                    "suggestions": self._format_suggestions({"intent": "product_query", "entities": entities, "lang": lang})
                })
                return response

            # Format Products
            formatted = []
            for p in products:
                try:
                    category_doc = self.categories_collection.find_one({"_id": p["category"]}, {"name": 1})
                    artisan_doc = self.users_collection.find_one({"_id": p["artisan"]}, {"name": 1})
                    if not category_doc or not artisan_doc:
                        logger.warning(f"Missing category or artisan for product: {p['_id']}")
                        continue

                    subcategories = []
                    for sub_id in p.get("subcategories", []):
                        sub_doc = self.subcategories_collection.find_one({"_id": sub_id}, {"name": 1})
                        if sub_doc:
                            subcategories.append(sub_doc["name"])

                    formatted.append({
                        "title": p["title"],
                        "description": p["description"],
                        "price": float(p["price"]),
                        "priceAfterDiscount": float(p.get("priceAfterDiscount", p["price"])),
                        "image": p["imageCover"],
                        "colors": p.get("colors", []),
                        "size": p.get("size", ""),
                        "weight": float(p.get("weight", 0.0)),
                        "rating": float(p.get("ratingsAverage", 0.0)),
                        "category": category_doc["name"],
                        "subcategories": subcategories,
                        "artisan": artisan_doc["name"],
                        "link": f"https://moderncsis.sytes.net/handmade/product/{p['_id']}"
                    })
                except Exception as e:
                    logger.warning(f"Error formatting product {p['_id']}: {str(e)}")
                    continue

            if not formatted:
                logger.debug("No products formatted successfully")
                response.update({
                    "status": "no_results",
                    "response": self.responses[lang]["no_results"][0],
                    "products": [],
                    "suggestions": self._format_suggestions({"intent": "product_query", "entities": entities, "lang": lang})
                })
                return response

            response["products"] = formatted
            response["response"] = self.responses[lang]["product_query"][0].format(count=len(formatted))
            logger.debug(f"Returning {len(formatted)} products")
            return response

        except Exception as e:
            logger.error(f"Error in product query handling: {str(e)}")
            return {
                "status": "no_results",
                "response": self.responses[lang]["no_results"][0],
                "products": [],
                "suggestions": self._format_suggestions({"intent": "product_query", "entities": entities, "lang": lang})
            }

    def _handle_filter(self, entities: Dict, lang: str) -> Dict:
        """Handle filter intent with structured MongoDB query."""
        try:
            filter_query = {"$and": []}

            # Price Range
            if entities.get("price_range"):
                min_price, max_price = entities["price_range"]
                if max_price == float('inf'):
                    max_price = 1e6
                if min_price > max_price:
                    min_price, max_price = max_price, min_price
                filter_query["$and"].append({
                    "$or": [
                        {"price": {"$gte": min_price, "$lte": max_price}},
                        {"priceAfterDiscount": {"$gte": min_price, "$lte": max_price}}
                    ]
                })
                logger.debug(f"Price filter: min={min_price}, max={max_price}")

            # Rating
            if entities.get("rating") is not None:
                filter_query["$and"].append({"ratingsAverage": {"$gte": entities["rating"]}})
                logger.debug(f"Rating filter: >= {entities['rating']}")

            # Categories
            if entities.get("categories"):
                category_ids = [
                    c["_id"] for c in self.categories_collection.find(
                        {"name": {"$regex": "^" + "|".join([re.escape(c) for c in entities["categories"]]) + "$", "$options": "i"}},
                        {"_id": 1}
                    )
                ]
                if category_ids:
                    filter_query["$and"].append({"category": {"$in": category_ids}})
                    logger.debug(f"Category IDs: {category_ids}")
                else:
                    logger.warning(f"No category IDs found for: {entities['categories']}")
                    available_categories = [c["name"] for c in self.categories_collection.find({}, {"name": 1})]
                    logger.debug(f"Available categories: {available_categories}")
                    return {
                        "status": "no_results",
                        "response": self.responses[lang]["filter"]["no_results"],
                        "products": [],
                        "suggestions": self._format_suggestions({"intent": "filter", "entities": entities, "lang": lang})
                    }

            # Subcategories
            if entities.get("subcategories"):
                subcat_names = []
                for s in entities["subcategories"]:
                    if isinstance(s, (tuple, list)) and len(s) >= 1:
                        subcat_names.append(s[0])
                    elif isinstance(s, str):
                        subcat_names.append(s)
                if subcat_names:
                    subcat_ids = [
                        s["_id"] for s in self.subcategories_collection.find(
                            {"name": {"$regex": "^" + "|".join([re.escape(n) for n in subcat_names]) + "$", "$options": "i"}},
                            {"_id": 1}
                        )
                    ]
                    if subcat_ids:
                        filter_query["$and"].append({"subcategories": {"$in": subcat_ids}})
                        logger.debug(f"Subcategory IDs: {subcat_ids}")

            # Artisans
            if entities.get("artisans"):
                artisan_ids = [
                    a["_id"] for a in self.users_collection.find(
                        {"name": {"$regex": "^" + "|".join([re.escape(a) for a in entities["artisans"]]) + "$", "$options": "i"}, "role": "artisan"},
                        {"_id": 1}
                    )
                ]
                if artisan_ids:
                    filter_query["$and"].append({"artisan": {"$in": artisan_ids}})
                    logger.debug(f"Artisan IDs: {artisan_ids}")

            # Colors
            if entities.get("colors"):
                color_filters = [
                    {"colors": {"$regex": re.escape(c), "$options": "i"}} for c in entities["colors"]
                ]
                filter_query["$and"].append({"$or": color_filters})
                logger.debug(f"Colors filter: {entities['colors']}")

            # Size and Weights
            for key in ["size", "weights"]:
                if entities.get(key):
                    if key == "size":
                        size_query = {"$regex": "^" + "|".join([re.escape(v) for v in entities[key]]) + "$", "$options": "i"}
                        filter_query["$and"].append({key: size_query})
                    else:
                        filter_query["$and"].append({key: {"$in": entities[key]}})
                    logger.debug(f"{key.capitalize()} filter: {entities[key]}")

            # Simplify query
            if not filter_query["$and"]:
                filter_query = {}
            logger.debug(f"Final filter query: {filter_query}")

            # Execute Query
            products = list(self.products_collection.find(filter_query, {
                "title": 1, "description": 1, "price": 1, "priceAfterDiscount": 1,
                "imageCover": 1, "colors": 1, "size": 1, "weight": 1, "ratingsAverage": 1,
                "category": 1, "subcategories": 1, "artisan": 1
            }).limit(5))

            if not products:
                logger.debug("No products found")
                return {
                    "status": "no_results",
                    "response": self.responses[lang]["filter"]["no_results"],
                    "products": [],
                    "suggestions": self._format_suggestions({"intent": "filter", "entities": entities, "lang": lang})
                }

            # Format Products
            formatted = []
            for p in products:
                try:
                    category_doc = self.categories_collection.find_one({"_id": p["category"]}, {"name": 1})
                    artisan_doc = self.users_collection.find_one({"_id": p["artisan"]}, {"name": 1})
                    if not category_doc or not artisan_doc:
                        logger.warning(f"Missing category or artisan for product: {p['_id']}")
                        continue

                    subcategories = []
                    for sub_id in p.get("subcategories", []):
                        sub_doc = self.subcategories_collection.find_one({"_id": sub_id}, {"name": 1})
                        if sub_doc:
                            subcategories.append(sub_doc["name"])

                    formatted.append({
                        "title": p["title"],
                        "description": p["description"],
                        "price": float(p["price"]),
                        "priceAfterDiscount": float(p.get("priceAfterDiscount", p["price"])),
                        "image": p["imageCover"],
                        "colors": p.get("colors", []),
                        "size": p.get("size", ""),
                        "weight": float(p.get("weight", 0.0)),
                        "rating": float(p.get("ratingsAverage", 0.0)),
                        "category": category_doc["name"],
                        "subcategories": subcategories,
                        "artisan": artisan_doc["name"],
                        "link": f"https://moderncsis.sytes.net/handmade/product/{p['_id']}"
                    })
                except Exception as e:
                    logger.warning(f"Error formatting product {p['_id']}: {str(e)}")
                    continue

            if not formatted:
                logger.debug("No products formatted successfully")
                return {
                    "status": "no_results",
                    "response": self.responses[lang]["filter"]["no_results"],
                    "products": [],
                    "suggestions": self._format_suggestions({"intent": "filter", "entities": entities, "lang": lang})
                }

            logger.debug(f"Returning {len(formatted)} products")
            return {
                "status": "success",
                "response": self.responses[lang]["filter"]["success"].format(count=len(formatted)),
                "products": formatted,
                "suggestions": self._format_suggestions({"intent": "filter", "entities": entities, "lang": lang})
            }

        except Exception as e:
            logger.error(f"Error in filter intent: {str(e)}")
            return {
                "status": "no_results",
                "response": self.responses[lang]["filter"]["no_results"],
                "products": [],
                "suggestions": self._format_suggestions({"intent": "filter", "entities": entities, "lang": lang})
            }
    
    def _handle_price_query(self, entities: Dict, lang: str) -> Dict:
        """Handle price queries with improved error handling"""
        try:
            # Always safely initialize price range
            price_range = entities.get("price_range")
            if price_range:
                min_price, max_price = price_range
            else:
                min_price, max_price = 0, float('inf')

            # Build price query
            query = {
                "$or": [
                    {"price": {"$gte": min_price, "$lte": max_price if max_price != float('inf') else 1000000}},
                    {"priceAfterDiscount": {"$gte": min_price, "$lte": max_price if max_price != float('inf') else 1000000}}
                ]
            }

            # Execute query with limit
            products = list(self.products_collection.find(query).limit(5))

            # Format products for response
            formatted_products = []
            for product in products:
                try:
                    cleaned = self._format_product(product)
                    if cleaned:
                        formatted = self._format_product_response(cleaned, lang)
                        if formatted:
                            formatted_products.append(formatted)
                except Exception as e:
                    logger.warning(f"Error formatting product for price query: {str(e)}")
                    continue
                
            if formatted_products:
                if max_price == float('inf'):
                    response = self.responses[lang]["price_range_products"][0].format(
                        min_price=min_price,
                        max_price="∞",
                        currency="EGP"
                    )
                else:
                    response = self.responses[lang]["price_range_products"][0].format(
                        min_price=min_price,
                        max_price=max_price,
                        currency="EGP"
                    )

                return {
                    "status": "success",
                    "response": response,
                    "products": formatted_products,
                    "suggestions": self._format_suggestions(lang, intent="price", entities=entities)
                }
            else:
                return {
                    "status": "no_results",
                    "response": self.responses[lang]["no_products_price"][0].format(
                        min_price=min_price,
                        max_price=max_price if max_price != float('inf') else "∞",
                        currency="EGP"
                    ),
                    "suggestions": self._format_suggestions(lang, intent="price", entities=entities)
                }

        except Exception as e:
            logger.error(f"Error in price query handling: {str(e)}")
            return self._handle_error(e, lang)

    def _handle_category_query(self, entities: Dict, lang: str) -> Dict:
        """Handle category-related queries with structured MongoDB query."""
        try:
            query = {}

            # Categories
            if entities.get("categories"):
                category_ids = [
                    c["_id"] for c in self.categories_collection.find(
                        {"name": {"$regex": "^" + "|".join([re.escape(c) for c in entities["categories"]]) + "$", "$options": "i"}},
                        {"_id": 1}
                    )
                ]
                if category_ids:
                    query["category"] = {"$in": category_ids}
                    logger.debug(f"Category IDs: {category_ids}")
                else:
                    logger.warning(f"No category IDs found for: {entities['categories']}")
                    available_categories = [c["name"] for c in self.categories_collection.find({}, {"name": 1})]
                    logger.debug(f"Available categories: {available_categories}")
                    return {
                        "status": "no_results",
                        "response": self.responses[lang]["no_results"][0],
                        "products": [],
                        "suggestions": self._format_suggestions({"intent": "category_query", "entities": entities, "lang": lang})
                    }

            # Subcategories
            if entities.get("subcategories"):
                subcat_names = []
                for s in entities["subcategories"]:
                    if isinstance(s, (tuple, list)) and len(s) >= 1:
                        subcat_names.append(s[0])
                    elif isinstance(s, str):
                        subcat_names.append(s)
                if subcat_names:
                    subcat_ids = [
                        s["_id"] for s in self.subcategories_collection.find(
                            {"name": {"$regex": "^" + "|".join([re.escape(n) for n in subcat_names]) + "$", "$options": "i"}},
                            {"_id": 1}
                        )
                    ]
                    if subcat_ids:
                        query["subcategories"] = {"$in": subcat_ids}
                        logger.debug(f"Subcategory IDs: {subcat_ids}")

            # Colors
            if entities.get("colors"):
                color_filters = [
                    {"colors": {"$regex": re.escape(c), "$options": "i"}} for c in entities["colors"]
                ]
                query["$or"] = color_filters
                logger.debug(f"Colors filter: {entities['colors']}")

            # Size and Weights
            for key in ["size", "weights"]:
                if entities.get(key):
                    if key == "size":
                        size_query = {"$regex": "^" + "|".join([re.escape(v) for v in entities[key]]) + "$", "$options": "i"}
                        query[key] = size_query
                    else:
                        query[key] = {"$in": entities[key]}
                    logger.debug(f"{key.capitalize()} filter: {entities[key]}")

            # Artisans
            if entities.get("artisans"):
                artisan_ids = [
                    a["_id"] for a in self.users_collection.find(
                        {"name": {"$regex": "^" + "|".join([re.escape(a) for a in entities["artisans"]]) + "$", "$options": "i"}, "role": "artisan"},
                        {"_id": 1}
                    )
                ]
                if artisan_ids:
                    query["artisan"] = {"$in": artisan_ids}
                    logger.debug(f"Artisan IDs: {artisan_ids}")

            logger.debug(f"Category query: {query}")

            # Execute Query
            products = list(self.products_collection.find(query, {
                "title": 1, "description": 1, "price": 1, "priceAfterDiscount": 1,
                "imageCover": 1, "colors": 1, "size": 1, "weight": 1, "ratingsAverage": 1,
                "category": 1, "subcategories": 1, "artisan": 1
            }).limit(5))

            if not products:
                logger.debug("No products found")
                return {
                    "status": "no_results",
                    "response": self.responses[lang]["no_results"][0],
                    "products": [],
                    "suggestions": self._format_suggestions({"intent": "category_query", "entities": entities, "lang": lang})
                }

            # Format Products
            formatted = []
            for p in products:
                try:
                    category_doc = self.categories_collection.find_one({"_id": p["category"]}, {"name": 1})
                    artisan_doc = self.users_collection.find_one({"_id": p["artisan"]}, {"name": 1})
                    if not category_doc or not artisan_doc:
                        logger.warning(f"Missing category or artisan for product: {p['_id']}")
                        continue

                    subcategories = []
                    for sub_id in p.get("subcategories", []):
                        sub_doc = self.subcategories_collection.find_one({"_id": sub_id}, {"name": 1})
                        if sub_doc:
                            subcategories.append(sub_doc["name"])

                    formatted.append({
                        "title": p["title"],
                        "description": p["description"],
                        "price": float(p["price"]),
                        "priceAfterDiscount": float(p.get("priceAfterDiscount", p["price"])),
                        "image": p["imageCover"],
                        "colors": p.get("colors", []),
                        "size": p.get("size", ""),
                        "weight": float(p.get("weight", 0.0)),
                        "rating": float(p.get("ratingsAverage", 0.0)),
                        "category": category_doc["name"],
                        "subcategories": subcategories,
                        "artisan": artisan_doc["name"],
                        "link": f"https://moderncsis.sytes.net/handmade/product/{p['_id']}"
                    })
                except Exception as e:
                    logger.warning(f"Error formatting product {p['_id']}: {str(e)}")
                    continue

            if not formatted:
                logger.debug("No products formatted successfully")
                return {
                    "status": "no_results",
                    "response": self.responses[lang]["no_results"][0],
                    "products": [],
                    "suggestions": self._format_suggestions({"intent": "category_query", "entities": entities, "lang": lang})
                }

            logger.debug(f"Returning {len(formatted)} products")
            return {
                "status": "success",
                "response": self.responses[lang]["product_query"][0]["success"].format(count=len(formatted)),
                "products": formatted,
                "suggestions": self._format_suggestions({"intent": "category_query", "entities": entities, "lang": lang})
            }

        except Exception as e:
            logger.error(f"Error handling category query: {str(e)}")
            return {
                "status": "no_results",
                "response": self.responses[lang]["no_results"][0],
                "products": [],
                "suggestions": self._format_suggestions({"intent": "category_query", "entities": entities, "lang": lang})
            }
    
    def _format_filter_summary(self, entities: Dict, lang: str, currency: str = "EGP") -> str:
        """Format a summary of applied filters"""
        filter_info = []

        # Format category filters
        if entities.get("categories") or entities.get("subcategories"):
            cats = entities.get("categories", []) + [sub[0] for sub in entities.get("subcategories", [])]
            if lang == "en":
                filter_info.append(f"Category: {', '.join(cats)}")
            else:
                filter_info.append(f"الفئة: {', '.join(cats)}")

        # Format price filters
        if entities.get("price_range"):
            min_price, max_price = entities["price_range"]
            if lang == "en":
                if max_price == float('inf'):
                    filter_info.append(f"Price: Above {min_price} {currency}")
                else:
                    filter_info.append(f"Price: {min_price}-{max_price} {currency}")
            else:
                if max_price == float('inf'):
                    filter_info.append(f"السعر: فوق {min_price} {currency}")
                else:
                    filter_info.append(f"السعر: {min_price}-{max_price} {currency}")

        # Format rating filters
        if entities.get("rating") is not None:
            if lang == "en":
                filter_info.append(f"Rating: {entities['rating']}+ stars")
            else:
                filter_info.append(f"التقييم: {entities['rating']}+ نجوم")

        # Format other filters
        for key, label in [
            ("colors", "Colors" if lang == "en" else "الألوان"),
            ("size", "Size" if lang == "en" else "الحجم"),
            ("weight", "Weight" if lang == "en" else "الوزن"),
            ("locations", "Location" if lang == "en" else "الموقع"),
            ("artisans", "Artisan" if lang == "en" else "الحرفي"),
            ("materials", "Materials" if lang == "en" else "المواد")
        ]:
            if entities.get(key):
                filter_info.append(f"{label}: {', '.join(entities[key])}")

        return " | ".join(filter_info) if filter_info else ""

    def _get_recommendations(self, user_id=None):
        try:
            if user_id:
                response = requests.get(f"{self.recommendation_service_url}/recommend/{user_id}")
            else:
                response = requests.get(f"{self.recommendation_service_url}/popular")
            
            if response.status_code == 200:
                recommendations = response.json()
                
                valid_recommendations = []
                for rec in recommendations:
                    product = self.products_collection.find_one({"_id": ObjectId(rec.get('_id'))})
                    if product:
                        valid_recommendations.append({
                            '_id': str(product['_id']),
                            'title': product['title'],
                            'price': float(product['price']),
                            'priceAfterDiscount': float(product.get('priceAfterDiscount', product['price'])),
                            'ratingsAverage': float(product['ratingsAverage']),
                            'ratingsQuantity': int(product['ratingsQuantity']),
                            'category': product['category']['name'] if isinstance(product.get('category'), dict) else None,
                            'artisan': f"{product['artisan']['name']} ({product['artisan']['_id']})" if isinstance(product.get('artisan'), dict) else None,
                            'description': product['description']
                        })
                return valid_recommendations
            else:
                logger.error(f"Error getting recommendations: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error calling recommendation service: {str(e)}")
            return []

    def _format_product_details(self, product, lang):
        """Format product details with emojis and proper structure"""
        details = []
        
        # Format size with emoji
        if product.get('size'):
            size_emoji = "📏"
            details.append(f"{size_emoji} {product['size']}")
        
        # Format colors with emoji
        if product.get('colors'):
            color_emoji = "🎨"
            details.append(f"{color_emoji} {', '.join(product['colors'])}")
        
        # Format weight with emoji
        if product.get('weight'):
            weight_emoji = "⚖️"
            details.append(f"{weight_emoji} {product['weight']} kg")
        
        # Format location with emoji
        if product.get('location'):
            location_emoji = "📍"
            details.append(f"{location_emoji} {product['location']}")
        
        # Format artisan with emoji
        if product.get('artisan', {}).get('name'):
            artisan_emoji = "👨‍🎨"
            details.append(f"{artisan_emoji} {product['artisan']['name']}")
        
        return " | ".join(details) if details else ""

    def _format_price(self, product, lang):
        """Format price with discount information if available"""
        price = round(product.get('priceAfterDiscount', product['price']), 2)
        original_price = round(product.get('price', 0), 2)
        currency = product.get('currency', 'EGP')

        if product.get('priceAfterDiscount') and price < original_price:
            discount = ((original_price - price) / original_price) * 100
            if lang == "en":
                return f"~~{original_price} {currency}~~ {price} {currency} (Save {discount:.0f}%)"
            else:
                return f"~~{original_price} {currency}~~ {price} {currency} (وفر {discount:.0f}%)"

        return f"{price} {currency}"

    def _format_rating(self, product, lang):
        """Format rating with stars and review count"""
        avg = product.get('ratingsAverage')
        qty = product.get('ratingsQuantity')

        if not (avg and qty):
            return ""

        stars = "⭐" * int(avg)
        if avg % 1 >= 0.5:
            stars += "½"

        if lang == "en":
            return f"{stars} ({avg:.1f} from {qty} reviews)"
        else:
            return f"{stars} ({avg:.1f} من {qty} تقييم)"

    def _format_product_response(self, product: Dict, lang: str) -> Optional[Dict]:
        """Format a single product response with better structure."""
        try:
            if not product or "_id" not in product:
                return None

            product_id = str(product["_id"])
            product_url = f"https://moderncsis.sytes.net/handmade/product/{product_id}"

            # Resolve subcategory names if ObjectIds
            subcategories = product.get("subcategories", [])
            if subcategories and isinstance(subcategories[0], str) and ObjectId.is_valid(subcategories[0]):
                subcategories = [
                    sub["name"] for sub in self.subcategories_collection.find(
                        {"_id": {"$in": [ObjectId(sc) for sc in subcategories]}}, {"name": 1}
                    )
                ]

            return {
                "title": product.get("title", ""),
                "description": product.get("description", ""),
                "price": float(product.get("price", 0)),
                "priceAfterDiscount": float(product.get("priceAfterDiscount", product.get("price", 0))),
                "rating": float(product.get("ratingsAverage", 0)),
                "weight": float(product.get("weight", 0.0)),
                "category": str(product.get("category", "")),
                "subcategories": subcategories,
                "artisan": str(product.get("artisan", "")),
                "colors": product.get("colors", []),
                "size": product.get("size", ""),
                "image": product.get("imageCover", ""),
                "link": product_url,
                "message": product.get("message", "")
            }
        except Exception as e:
            logger.error(f"Error formatting product response: {str(e)}")
            return None

    def _handle_greeting(self, lang: str) -> Dict:
        """Handle greeting intent with product suggestions"""
        try:
            # Get a random greeting response
            greeting = random.choice(self.responses[lang]['greeting'])
            
            return {
                "status": "success",
                "response": greeting,
                "suggestions": self._get_sample_queries(lang)
            }
        except Exception as e:
            logger.error(f"Error handling greeting: {str(e)}")
            return self._handle_error(e, lang)
    
    def _handle_recommendation(self, lang: str) -> Dict:
        if not self.recommendation_service_url:
            return {
                "status": "error",
                "response": self.responses[lang]["error"][0]
            }

        try:
            res = requests.get(self.recommendation_service_url)
            res.raise_for_status()
            data = res.json()
            products = data.get("products", [])

            formatted = [
                self._format_product_response(self._format_product(p), lang)
                for p in products if p
            ]

            return {
                "status": "success",
                "response": self.responses[lang]["popular"][0],
                "products": [f for f in formatted if f]  # Filter out None
            }
        except Exception as e:
            logger.error(f"Recommendation API error: {str(e)}")
            return self._handle_error(e, lang)
    
    def get_artisans(self):
        """Get list of artisans from users collection"""
        try:
            artisans = list(self.users_collection.find(
                {"role": "artisan"},
                {"name": 1, "email": 1, "profile_picture": 1, "addresses": 1}
            ))
            return artisans
        except Exception as e:
            logger.error(f"Error getting artisans: {str(e)}")
            return []

    def get_locations(self):
        """Get list of unique locations from user addresses and product stats"""
        try:
            location_stats = {}

            # Step 1: Get all artisans with addresses
            artisan_cursor = self.users_collection.find(
                {"role": "artisan", "addresses": {"$exists": True, "$ne": []}},
                {"addresses.city": 1}
            )
            artisan_map = {}  # _id -> city list

            for artisan in artisan_cursor:
                artisan_id = str(artisan["_id"])
                artisan_map[artisan_id] = []
                for address in artisan.get("addresses", []):
                    if address.get("city"):
                        city = address["city"].strip()
                        artisan_map[artisan_id].append(city)
                        if city not in location_stats:
                            location_stats[city] = {
                                'name': city,
                                'product_count': 0,
                                'min_price': float('inf'),
                                'max_price': 0,
                                'total_price': 0,
                                'total_rating': 0,
                                'rating_count': 0,
                                'artisan_ids': set()
                            }
                        location_stats[city]["artisan_ids"].add(artisan_id)

            # Step 2: Aggregate product statistics
            for product in self.products_collection.find(
                {"artisan": {"$exists": True}}, {"artisan": 1, "price": 1, "ratingsAverage": 1}
            ):
                artisan_id = str(product["artisan"])
                artisan_cities = artisan_map.get(artisan_id, [])

                for city in artisan_cities:
                    stats = location_stats[city]
                    stats['product_count'] += 1
                    stats['min_price'] = min(stats['min_price'], product.get('price', 0))
                    stats['max_price'] = max(stats['max_price'], product.get('price', 0))
                    stats['total_price'] += product.get('price', 0)
                    if 'ratingsAverage' in product:
                        stats['total_rating'] += product['ratingsAverage']
                        stats['rating_count'] += 1

            # Step 3: Format results
            locations = []
            for city, stats in location_stats.items():
                if stats['product_count'] > 0:
                    locations.append({
                        'name': city,
                        'product_count': stats['product_count'],
                        'min_price': 0 if stats['min_price'] == float('inf') else stats['min_price'],
                        'max_price': stats['max_price'],
                        'average_price': round(stats['total_price'] / stats['product_count'], 2),
                        'average_rating': round(stats['total_rating'] / stats['rating_count'], 2) if stats['rating_count'] > 0 else 0,
                        'artisan_count': len(stats['artisan_ids'])
                    })

            locations.sort(key=lambda x: x['name'])
            return locations

        except Exception as e:
            logger.error(f"Error in get_locations: {str(e)}")
            return []

    def _handle_help(self, lang: str) -> Dict:
        """Handle help intent with context-aware suggestions"""
        return {
            "status": "help",
            "response": self.responses[lang]["help"][0],
            "suggestions": self._format_suggestions(lang, intent="help", entities={})
        }

    def _handle_feedback(self, lang: str) -> Dict:
        """Handle feedback intent"""
        return {
            "status": "feedback",
            "response": self.responses[lang]["feedback"][0],
            "suggestions": self._format_suggestions(lang, intent="feedback", entities={})
        }

    # Transliteration mapping for common Arabic terms
    ARABIC_TRANSLITERATION = {
        # Numbers
        '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
        '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
        
        # Common letters
        'a': 'ا', 'b': 'ب', 't': 'ت', 'th': 'ث', 'g': 'ج',
        '7': 'ح', 'kh': 'خ', 'd': 'د', 'z': 'ز', 'r': 'ر',
        's': 'س', 'sh': 'ش', '9': 'ص', '6': 'ض',
        '3': 'ع', 'gh': 'غ', 'f': 'ف', 'q': 'ق',
        'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن', 'h': 'ه',
        'w': 'و', 'y': 'ي', 'p': 'ب', 'v': 'ف', 'e': 'ي',
        'o': 'و', 'u': 'و',
        
        # Common words
        'ana': 'انا', 'enta': 'انت', 'enti': 'انتي', 'howa': 'هو',
        'heya': 'هي', 'ehna': 'احنا', 'ento': 'انتو', 'homma': 'هما',
        'ayez': 'عايز', '3ayez': 'عايز', '3awez': 'عاوز', '7abiby': 'حبيبي',
        'm3ana': 'معانا', 'm3akom': 'معاكم', 'm3ahom': 'معاهم',
        
        # Category and subcategory transliterations
        'akseswarat': 'اكسسوارات', 'seramik': 'سيراميك', 'fokhar': 'فخار',
        'zujaj': 'زجاج', 'jeld': 'جلد', 'rukham': 'رخام', 'khshb': 'خشب',
        'dikour': 'ديكور', 'awalim': 'أواني', 'sharib': 'شرب', 'ta3am': 'طعام',
        'tabkh': 'طبخ', 'mashroub': 'مشروب', 'ma2ida': 'مائدة', 'atbak': 'أطباق',
        'zana': 'زينة', 'manzili': 'منزلي', 'akseswar': 'اكسسوار'
    }

    def replace_transliterated_words(self, text: str) -> str:
        """
        Replace transliterated Arabic words with their Arabic equivalents.
        Handles both single letters and common words.
        """
        try:
            if not text:
                return ""
                
            # Convert to string if not already
            text = str(text)
            
            # First replace common words (longer matches first)
            for translit, arabic in sorted(self.ARABIC_TRANSLITERATION.items(), key=lambda x: len(x[0]), reverse=True):
                if len(translit) > 1:  # Only replace words, not single letters
                    text = re.sub(r'\b' + re.escape(translit) + r'\b', arabic, text, flags=re.IGNORECASE)
            
            # Then replace single letters
            for translit, arabic in self.ARABIC_TRANSLITERATION.items():
                if len(translit) == 1:  # Only replace single letters
                    text = text.replace(translit, arabic)
            
            return text
            
        except Exception as e:
            logger.error(f"Error in transliteration replacement: {str(e)}")
            return text

    def split_mixed_script_tokens(self, text: str) -> str:
        """Split tokens mixing Arabic and Latin scripts, handling numbers and special characters."""
        try:
            if not text:
                return ""
            
            text = str(text)
            
            # Enhanced pattern to match mixed script tokens, including numbers and special characters
            mixed_pattern = re.compile(
                r'([\u0600-\u06FF]+)([a-zA-Z0-9]+)|([a-zA-Z0-9]+)([\u0600-\u06FF]+)'
            )
            
            # Replace mixed tokens with space-separated versions
            text = mixed_pattern.sub(
                lambda m: f"{m.group(1) or m.group(3)} {m.group(2) or m.group(4)}", text
            )
            
            # Handle glued cases with special characters
            text = re.sub(r'([\u0600-\u06FF]+)([a-zA-Z0-9]+)', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z0-9]+)([\u0600-\u06FF]+)', r'\1 \2', text)
            
            # Clean up extra spaces
            text = ' '.join(text.split())
            
            return text
        
        except Exception as e:
            logger.error(f"Error in mixed script token splitting: {str(e)}")
            return text

# Custom JSON encoder setup
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)

app.json_encoder = JSONEncoder

# Chatbot service initialization
chatbot_instance = None

try:
    mongo_uri = os.getenv('MONGODB_URI')
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")

    recommendation_service_url = os.getenv('RECOMMENDATION_SERVICE_URL')
    if not recommendation_service_url:
        logger.warning("RECOMMENDATION_SERVICE_URL not set — recommendations will be disabled")

    chatbot_instance = ChatbotService(
        mongo_uri=mongo_uri,
        db_name="handMade",
        recommendation_service_url=recommendation_service_url
    )

    chatbot_instance.embedding_service._build_faiss_index()
    logger.info("Chatbot service and FAISS index initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize chatbot service: {str(e)}\n{traceback.format_exc()}")

app.chatbot = chatbot_instance

@app.route('/', methods=['GET'])
def index():
    if app.chatbot is None:
        return jsonify({
            "message": "Chatbot service failed to initialize.",
            "service": "chatbot-service",
            "status": "error"
        }), 500

    # Define all available endpoints in a cleaner format
    endpoints = {
        'root': '/',
        'chat': '/chat',
        'health': '/health',
        'api_info': '/api/info',
        'categories': '/api/categories',
        'artisans': '/api/artisans',
        'locations': '/api/locations',
        'debug_artisan_locations': '/api/debug/artisan-locations',
        'debug_products': '/api/debug/products'
    }

    # Get service status
    service_status = "running" if app.chatbot.is_mongo_connected() else "waiting"
    
    response = {
        "service": "chatbot-service",
        "status": service_status,
        "message": "Chatbot service is running. Database is connected." if service_status == "running" else "Chatbot service is running. Waiting for database setup.",
        "endpoints": endpoints,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body", "status": "error"}), 400

        text = data.get("text") or data.get("message", "")
        if not text.strip():
            return jsonify({"error": "Message cannot be empty", "status": "error"}), 400

        lang = data.get("lang")
        if not lang:
            try:
                lang = detect(text)
            except LangDetectException:
                lang = "en"
        if lang not in ["en", "ar"]:
            lang = "en"

        if not app.chatbot:
            return jsonify({"error": "Chatbot service is not initialized", "status": "error"}), 500

        # Classify intent
        result = app.chatbot._classify_intent(text, lang)
        intent = result["intent"]
        confidence = result["confidence"]
        entities = result["entities"]
        logger.debug(f"Classified intent: {intent}, confidence: {confidence}, entities: {entities}")

        # Route semantic_query to _safe_faiss_search
        if intent == "semantic_query":
            logger.debug(f"Routing semantic_query to _safe_faiss_search for message: {text}")
            response = app.chatbot._safe_faiss_search(text, lang, entities, intent)
        else:
            logger.debug(f"Routing intent {intent} to process_message for message: {text}")
            response = app.chatbot.process_message(text, lang, intent=intent, entities=entities)

        # Add request metadata for testing
        response.update({
            "request": {
                "text": text,
                "lang": lang,
                "timestamp": datetime.now().isoformat()
            }
        })
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "An unexpected error occurred. Please try again.",
            "status": "error",
            "details": str(e) if app.debug else None
        }), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    if app.chatbot is None:
        return jsonify({
            "error": "Chatbot service not initialized",
            "status": "error"
        }), 500
    
    try:
        # Check MongoDB connection
        mongo_status = "connected" if app.chatbot.is_mongo_connected() else "disconnected"
        
        # Check embedding service
        embedding_status = "loaded" if hasattr(app.chatbot, 'embedding_service') and app.chatbot.embedding_service.model is not None else "not loaded"
        
        # Check recommendation service
        try:
            if app.chatbot.recommendation_service_url:
                response = requests.get(f"{app.chatbot.recommendation_service_url}/health")
                rec_status = "connected" if response.status_code == 200 else "disconnected"
            else:
                rec_status = "not configured"
        except:
            rec_status = "disconnected"
        
        # Get data statistics
        stats = {
            "products_count": app.chatbot.products_collection.count_documents({}),
            "categories_count": app.chatbot.categories_collection.count_documents({}),
            "artisans_count": app.chatbot.users_collection.count_documents({"role": "artisan"}),
            "locations_count": len(app.chatbot.get_locations())
        }
        
        return jsonify({
            "service": "chatbot-service",
            "status": "healthy",
            "components": {
                "mongodb": mongo_status,
                "embedding_service": embedding_status,
                "recommendation_service": rec_status
            },
            "data_statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    if app.chatbot is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get product prices using aggregation
        price_stats = app.chatbot.products_collection.aggregate([
            {
                "$group": {
                    "_id": None,
                    "min_price": {"$min": "$price"},
                    "max_price": {"$max": "$price"}
                }
            }
        ])
        
        price_data = next(price_stats, {"min_price": 0, "max_price": 0})
        min_price = float(price_data.get('min_price', 0))
        max_price = float(price_data.get('max_price', 0))

        # Get counts using proper collection queries
        total_products = app.chatbot.products_collection.count_documents({})
        total_categories = app.chatbot.categories_collection.count_documents({})
        total_locations = len(app.chatbot.get_locations())
        total_artisans = app.chatbot.users_collection.count_documents({"role": "artisan"})

        return jsonify({
            'service': 'chatbot-service',
            'capabilities': {
                'intents': {
                    'greeting': 'Start a conversation',
                    'recommendation': 'Get product recommendations',
                    'price': 'Search by price range',
                    'category': 'Browse by category',
                    'location': 'Find products by location',
                    'artisan': 'Search by artisan',
                    'rating': 'Find highly-rated products',
                    'help': 'Get help and instructions'
                },
                'example_queries': [
                    'Show me products under 100 EGP',
                    'What categories do you have?',
                    'Show me products from Cairo',
                    'Who are your artisans?',
                    'What are your popular items?',
                    'Show me products in the pottery category'
                ],
                'price_ranges': {
                    'min': min_price,
                    'max': max_price,
                    'currency': 'EGP'
                },
                'total_products': total_products,
                'total_categories': total_categories,
                'total_locations': total_locations,
                'total_artisans': total_artisans
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/info endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve API info',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    if app.chatbot is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get category statistics with proper lookup
        category_stats = app.chatbot.products_collection.aggregate([
            {"$lookup": {
                "from": "categories",
                "localField": "category",
                "foreignField": "_id",
                "as": "category_info"
            }},
            {"$unwind": "$category_info"},
            {"$group": {
                "_id": "$category_info.name",
                "product_count": {"$sum": 1},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"},
                "avg_price": {"$avg": "$price"},
                "avg_rating": {"$avg": "$ratingsAverage"}
            }},
            {"$project": {
                "_id": 0,
                "name": "$_id",
                "product_count": 1,
                "min_price": 1,
                "max_price": 1,
                "average_price": {"$round": ["$avg_price", 2]},
                "average_rating": {"$round": ["$avg_rating", 2]}
            }},
            {"$sort": {"name": 1}}
        ])
        
        # Convert cursor to list
        categories = list(category_stats)
        
        return jsonify({
            'categories': categories,
            'total_categories': len(categories),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/categories endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get categories',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/artisans', methods=['GET'])
def get_artisans():
    if app.chatbot is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get artisan statistics with proper lookup
        artisan_stats = app.chatbot.products_collection.aggregate([
            {"$lookup": {
                "from": "users",
                "localField": "artisan",
                "foreignField": "_id",
                "as": "artisan_info"
            }},
            {"$unwind": "$artisan_info"},
            {"$match": {"artisan_info.role": "artisan"}},
            {"$group": {
                "_id": "$artisan_info.name",
                "artisan_id": {"$first": "$artisan"},
                "email": {"$first": "$artisan_info.email"},
                "profile_picture": {"$first": "$artisan_info.profile_picture"},
                "product_count": {"$sum": 1},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"},
                "avg_price": {"$avg": "$price"},
                "avg_rating": {"$avg": "$ratingsAverage"},
                "categories": {"$addToSet": "$category"}
            }},
            {"$lookup": {
                "from": "categories",
                "localField": "categories",
                "foreignField": "_id",
                "as": "category_info"
            }},
            {"$project": {
                "_id": 0,
                "name": "$_id",
                "artisan_id": 1,
                "email": 1,
                "profile_picture": 1,
                "product_count": 1,
                "min_price": 1,
                "max_price": 1,
                "average_price": {"$round": ["$avg_price", 2]},
                "average_rating": {"$round": ["$avg_rating", 2]},
                "categories": "$category_info.name"
            }},
            {"$sort": {"name": 1}}
        ])
        
        # Convert cursor to list
        artisans = list(artisan_stats)
        
        # Get location info for artisans
        for artisan in artisans:
            artisan_user = app.chatbot.users_collection.find_one(
                {"_id": ObjectId(artisan['artisan_id'])},
                {"addresses.city": 1}
            )
            if artisan_user and artisan_user.get('addresses'):
                artisan['location'] = artisan_user['addresses'][0].get('city')
            artisan['artisan_id'] = str(artisan['artisan_id'])  # Convert ObjectId to string
        
        return jsonify({
            'artisans': artisans,
            'total_artisans': len(artisans),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/artisans endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get artisans',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/debug/artisan-locations', methods=['GET'])
def debug_artisan_locations():
    if app.chatbot is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get all artisans with their products and locations
        artisans = []
        for user in app.chatbot.users_collection.find({"role": "artisan"}):
            # Convert user document to dict with string ObjectIds
            user_dict = {}
            for key, value in user.items():
                if isinstance(value, ObjectId):
                    user_dict[key] = str(value)
                elif key == 'addresses' and isinstance(value, list):
                    user_dict[key] = []
                    for addr in value:
                        addr_dict = {}
                        for k, v in addr.items():
                            addr_dict[k] = str(v) if isinstance(v, ObjectId) else v
                        user_dict[key].append(addr_dict)
                else:
                    user_dict[key] = value
            
            # Get products for this artisan
            products = []
            for product in app.chatbot.products_collection.find(
                {"artisan": user['_id']},
                {"_id": 1, "title": 1, "price": 1, "ratingsAverage": 1}
            ):
                product_dict = {
                    '_id': str(product['_id']),
                    'title': product.get('title'),
                    'price': product.get('price'),
                    'ratingsAverage': product.get('ratingsAverage')
                }
                products.append(product_dict)
            
            artisan = {
                'id': str(user['_id']),
                'name': user.get('name'),
                'email': user.get('email'),
                'addresses': user_dict.get('addresses', []),
                'products': products
            }
            
            artisans.append(artisan)
        
        return jsonify({
            'artisans': artisans,
            'total_artisans': len(artisans),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/debug/artisan-locations endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get artisan locations debug info',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    if app.chatbot is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get all locations from user addresses
        location_stats = {}
        
        # First, get all artisans with their addresses
        artisans = app.chatbot.users_collection.find(
            {"role": "artisan", "addresses": {"$exists": True, "$ne": []}},
            {"addresses.city": 1, "_id": 1}
        )
        
        # Initialize location stats for each city
        for artisan in artisans:
            for address in artisan.get('addresses', []):
                if address.get('city'):
                    city = address['city'].strip()
                    if city not in location_stats:
                        location_stats[city] = {
                            'name': city,
                            'product_count': 0,
                            'min_price': float('inf'),
                            'max_price': 0,
                            'total_price': 0,
                            'total_rating': 0,
                            'rating_count': 0,
                            'artisan_ids': set()
                        }
                    location_stats[city]['artisan_ids'].add(str(artisan['_id']))

        # Get product statistics for each location
        for product in app.chatbot.products_collection.find(
            {"artisan._id": {"$exists": True}},
            {"artisan._id": 1, "price": 1, "ratingsAverage": 1}
        ):
            # Get artisan's location
            artisan_id = product['artisan']['_id']
            artisan = app.chatbot.users_collection.find_one(
                {"_id": ObjectId(artisan_id)},
                {"addresses.city": 1}
            )
            if artisan and artisan.get('addresses'):
                for address in artisan['addresses']:
                    if address.get('city'):
                        city = address['city'].strip()
                        if city in location_stats:
                            stats = location_stats[city]
                            stats['product_count'] += 1
                            stats['min_price'] = min(stats['min_price'], product['price'])
                            stats['max_price'] = max(stats['max_price'], product['price'])
                            stats['total_price'] += product['price']
                            if product.get('ratingsAverage'):
                                stats['total_rating'] += product['ratingsAverage']
                                stats['rating_count'] += 1

        # Convert stats to list and calculate averages
        locations = []
        for city, stats in location_stats.items():
            location = {
                'name': city,
                'product_count': stats['product_count'],
                'min_price': 0 if stats['min_price'] == float('inf') else stats['min_price'],
                'max_price': stats['max_price'],
                'average_price': round(stats['total_price'] / stats['product_count'], 2) if stats['product_count'] > 0 else 0,
                'average_rating': round(stats['total_rating'] / stats['rating_count'], 2) if stats['rating_count'] > 0 else 0,
                'artisan_count': len(stats['artisan_ids'])
            }
            locations.append(location)

        # Sort locations by name
        locations.sort(key=lambda x: x['name'])
        
        return jsonify({
            'locations': locations,
            'total_locations': len(locations),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/locations endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get locations',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/debug/products', methods=['GET'])
def debug_products():
    try:
        # Get a sample product to inspect its structure
        sample_product = app.chatbot.products_collection.find_one()
        if sample_product:
            # Convert ObjectId to string
            if '_id' in sample_product:
                sample_product['_id'] = str(sample_product['_id'])
            
            # Also convert any nested ObjectIds
            def convert_objectids(obj):
                if isinstance(obj, dict):
                    return {k: convert_objectids(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_objectids(item) for item in obj]
                elif isinstance(obj, ObjectId):
                    return str(obj)
                return obj
            
            sample_product = convert_objectids(sample_product)
            
            return jsonify({
                'sample_product': sample_product,
                'total_products': app.chatbot.products_collection.count_documents({}),
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'message': 'No products found in the collection',
                'total_products': 0,
                'timestamp': datetime.now().isoformat()
            }), 200
    except Exception as e:
        logger.error(f"Error in /api/debug/products endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get product debug info',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port) 
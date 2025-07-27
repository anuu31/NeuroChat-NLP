import nltk
import random
import string
import warnings
import json
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import re
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class NLTKSetup:
    """Handles NLTK data downloads with error handling"""
    
    REQUIRED_PACKAGES = [
        'punkt',
        'stopwords', 
        'averaged_perceptron_tagger',
        'wordnet',
        'vader_lexicon',
        'punkt_tab'
    ]
    
    @classmethod
    def setup_nltk(cls) -> bool:
        """Download required NLTK packages"""
        try:
            print("🔄 Setting up NLP components...")
            
            for package in cls.REQUIRED_PACKAGES:
                try:
                    nltk.data.find(f'tokenizers/{package}')
                except LookupError:
                    try:
                        nltk.data.find(f'corpora/{package}')
                    except LookupError:
                        try:
                            nltk.data.find(f'taggers/{package}')
                        except LookupError:
                            print(f"📥 Downloading {package}...")
                            nltk.download(package, quiet=True)
            
            print("✅ NLP setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"NLTK setup failed: {e}")
            print("❌ NLP setup failed. Please check your internet connection.")
            return False

# Import NLTK components after setup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer

class AdvancedChatbot:
    """
    Advanced AI Chatbot with comprehensive NLP capabilities
    """
    
    def __init__(self):
        """Initialize chatbot with NLP components and knowledge base"""
        try:
            # Initialize NLP components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Conversation management
            self.conversation_history = deque(maxlen=100)  # Memory optimization
            self.user_context = {}
            self.session_start = datetime.now()
            
            # Performance metrics
            self.response_count = 0
            self.intent_accuracy = []
            
            # Load configurations
            self._load_intents()
            self._load_knowledge_base()
            self._load_personality_traits()
            
            logger.info("Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Chatbot initialization failed: {e}")
            raise
    
    def _load_intents(self):
        """Load intent patterns and responses"""
        self.intents = {
            'greeting': {
                'patterns': [
                    'hello', 'hi', 'hey', 'greetings', 'good morning', 
                    'good afternoon', 'good evening', 'namaste', 'hola'
                ],
                'responses': [
                    "Hello! I'm your AI assistant. How can I help you today? 🤖",
                    "Hi there! Ready to have an intelligent conversation? 💬",
                    "Greetings! I'm here to assist with any questions you have. ✨",
                    "Hey! What's on your mind today? Let's chat! 🌟"
                ],
                'context': 'social'
            },
            
            'farewell': {
                'patterns': [
                    'bye', 'goodbye', 'see you', 'farewell', 'quit', 'exit',
                    'take care', 'catch you later', 'signing off'
                ],
                'responses': [
                    "Goodbye! It was great chatting with you! Take care! 👋",
                    "See you later! Feel free to return anytime for more conversations! 🌟",
                    "Farewell! Thank you for the engaging conversation! 💫",
                    "Until next time! Keep learning and stay curious! 🚀"
                ],
                'context': 'social'
            },
            
            'gratitude': {
                'patterns': [
                    'thank you', 'thanks', 'appreciate it', 'grateful',
                    'much obliged', 'cheers', 'thank u'
                ],
                'responses': [
                    "You're absolutely welcome! Happy to help anytime! 😊",
                    "My pleasure! That's what I'm here for! 🤗",
                    "Glad I could assist! Feel free to ask anything else! ✨",
                    "You're very welcome! Helping you makes my circuits happy! 🤖"
                ],
                'context': 'social'
            },
            
            'identity': {
                'patterns': [
                    'who are you', 'what are you', 'your name', 'about yourself',
                    'introduce yourself', 'tell me about you'
                ],
                'responses': [
                    "I'm an advanced AI chatbot built with sophisticated NLP capabilities! 🧠",
                    "I'm your intelligent conversation partner, powered by advanced language processing! 💡",
                    "Call me ChatBot Pro - I'm designed to understand and respond naturally! 🚀",
                    "I'm an AI assistant created for the CODTECH internship, equipped with cutting-edge NLP! 🌟"
                ],
                'context': 'informational'
            },
            
            'capabilities': {
                'patterns': [
                    'what can you do', 'your capabilities', 'help me', 'functions',
                    'features', 'abilities', 'skills'
                ],
                'responses': [
                    "I can analyze sentiment, recognize intents, extract entities, have intelligent conversations, answer questions, and much more! 🔥",
                    "My capabilities include NLP processing, contextual understanding, knowledge retrieval, and engaging dialogue! 💪",
                    "I offer sentiment analysis, intent recognition, entity extraction, conversation memory, and intelligent responses! 🧠",
                    "I'm equipped with advanced language understanding, context awareness, and comprehensive knowledge base access! ⚡"
                ],
                'context': 'informational'
            },
            
            'time_query': {
                'patterns': [
                    'time', 'what time', 'current time', 'clock', 'hour'
                ],
                'responses': [
                    f"The current time is {datetime.now().strftime('%I:%M %p')} ⏰",
                    f"Right now it's {datetime.now().strftime('%H:%M:%S')} 🕐",
                    f"Current time: {datetime.now().strftime('%I:%M %p on %B %d, %Y')} 📅"
                ],
                'context': 'utility'
            },
            
            'date_query': {
                'patterns': [
                    'date', 'today', 'current date', 'what day', 'calendar'
                ],
                'responses': [
                    f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅",
                    f"Current date: {datetime.now().strftime('%Y-%m-%d')} 🗓️",
                    f"It's {datetime.now().strftime('%B %d, %Y')} - {datetime.now().strftime('%A')} 📆"
                ],
                'context': 'utility'
            },
            
            'humor': {
                'patterns': [
                    'joke', 'funny', 'humor', 'make me laugh', 'comedy',
                    'something funny', 'entertain me'
                ],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything! 😄",
                    "I told my computer a joke about UDP... but I'm not sure if it got it! 💻",
                    "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
                    "What's the object-oriented way to become wealthy? Inheritance! 💰",
                    "Why did the AI break up with the algorithm? It wasn't learning from the relationship! 🤖💔"
                ],
                'context': 'entertainment'
            },
            
            'motivation': {
                'patterns': [
                    'motivate', 'inspire', 'encourage', 'motivation', 'uplift',
                    'boost confidence', 'cheer up'
                ],
                'responses': [
                    "Believe in yourself! Every expert was once a beginner. You've got this! 💪",
                    "Success is not final, failure is not fatal: it's the courage to continue that counts! 🌟",
                    "The only way to do great work is to love what you do. Keep pushing forward! 🚀",
                    "Your potential is limitless! Every challenge is an opportunity to grow stronger! ✨",
                    "Remember: you're capable of amazing things. Trust the process! 🌈"
                ],
                'context': 'emotional_support'
            },
            
            'learning': {
                'patterns': [
                    'learn', 'study', 'education', 'knowledge', 'teach me',
                    'explain', 'understand', 'course'
                ],
                'responses': [
                    "Learning is a lifelong journey! What specific topic interests you? 📚",
                    "I'd love to help you learn! Knowledge is power - what would you like to explore? 🧠",
                    "Education is the key to unlocking potential! What subject can I assist with? 🎓",
                    "Great mindset! Continuous learning leads to growth. What's your learning goal today? 🌱"
                ],
                'context': 'educational'
            }
        }
    
    def _load_knowledge_base(self):
        """Load comprehensive knowledge base"""
        self.knowledge_base = {
            'artificial intelligence': {
                'definition': "Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think, learn, and make decisions like humans.",
                'applications': "AI is used in healthcare, finance, autonomous vehicles, recommendation systems, natural language processing, and robotics.",
                'types': "AI includes Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, and Robotics."
            },
            
            'machine learning': {
                'definition': "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
                'types': "Supervised Learning, Unsupervised Learning, and Reinforcement Learning are the main types.",
                'applications': "Used in predictive analytics, recommendation systems, image recognition, and fraud detection."
            },
            
            'natural language processing': {
                'definition': "NLP is a branch of AI that helps computers understand, interpret, and generate human language in a meaningful way.",
                'components': "Includes tokenization, parsing, sentiment analysis, named entity recognition, and language generation.",
                'applications': "Powers chatbots, translation services, sentiment analysis, and voice assistants."
            },
            
            'python programming': {
                'definition': "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                'features': "Object-oriented, extensive libraries, cross-platform compatibility, and active community support.",
                'applications': "Web development, data science, AI/ML, automation, and scientific computing."
            },
            
            'data science': {
                'definition': "Data Science combines statistics, programming, and domain expertise to extract insights from data.",
                'process': "Involves data collection, cleaning, analysis, visualization, and interpretation.",
                'tools': "Python, R, SQL, Tableau, Jupyter notebooks, and various ML libraries."
            },
            
            'nltk': {
                'definition': "NLTK (Natural Language Toolkit) is a comprehensive Python library for NLP tasks.",
                'features': "Tokenization, POS tagging, parsing, sentiment analysis, and corpus access.",
                'usage': "Widely used in academic research, prototyping, and educational projects."
            }
        }
    
    def _load_personality_traits(self):
        """Define chatbot personality characteristics"""
        self.personality = {
            'tone': 'friendly_professional',
            'humor_level': 'moderate',
            'formality': 'casual_professional',
            'empathy': 'high',
            'enthusiasm': 'high'
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """Advanced text preprocessing with optimization"""
        try:
            # Normalize text
            text = text.lower().strip()
            
            # Handle contractions
            contractions = {
                "won't": "will not", "can't": "cannot", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
            }
            
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
            
            # Remove excessive punctuation but keep sentence structure
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 1:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            return processed_tokens
            
        except Exception as e:
            logger.error(f"Text preprocessing error: {e}")
            return text.split()
    
    def analyze_sentiment_advanced(self, text: str) -> Dict:
        """Advanced sentiment analysis using VADER"""
        try:
            # VADER sentiment analysis
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Determine dominant sentiment
            compound = scores['compound']
            
            if compound >= 0.05:
                sentiment = 'positive'
                confidence = scores['pos']
            elif compound <= -0.05:
                sentiment = 'negative'  
                confidence = scores['neg']
            else:
                sentiment = 'neutral'
                confidence = scores['neu']
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'scores': scores,
                'compound': round(compound, 3)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'scores': {}, 'compound': 0}
    
    def extract_entities_advanced(self, text: str) -> List[Tuple[str, str]]:
        """Advanced named entity recognition"""
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity chunking
            entities = ne_chunk(pos_tags, binary=False)
            
            named_entities = []
            current_entity = []
            current_label = None
            
            for chunk in entities:
                if hasattr(chunk, 'label'):
                    if current_entity and current_label:
                        entity_text = ' '.join([token for token, pos in current_entity])
                        named_entities.append((entity_text, current_label))
                    
                    current_entity = chunk.leaves()
                    current_label = chunk.label()
                else:
                    if current_entity and current_label:
                        entity_text = ' '.join([token for token, pos in current_entity])
                        named_entities.append((entity_text, current_label))
                        current_entity = []
                        current_label = None
            
            # Handle last entity
            if current_entity and current_label:
                entity_text = ' '.join([token for token, pos in current_entity])
                named_entities.append((entity_text, current_label))
            
            return named_entities
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    def detect_intent_advanced(self, text: str) -> Tuple[Optional[str], float]:
        """Advanced intent detection with confidence scoring"""
        try:
            processed_text = ' '.join(self.preprocess_text(text))
            original_text = text.lower()
            
            best_intent = None
            best_score = 0.0
            
            for intent, data in self.intents.items():
                intent_score = 0.0
                pattern_matches = 0
                
                for pattern in data['patterns']:
                    # Exact match bonus
                    if pattern in original_text:
                        intent_score += len(pattern.split()) * 2
                        pattern_matches += 1
                    
                    # Partial match scoring
                    pattern_words = set(pattern.split())
                    text_words = set(original_text.split())
                    overlap = len(pattern_words.intersection(text_words))
                    
                    if overlap > 0:
                        intent_score += overlap * 0.5
                
                # Normalize score
                if pattern_matches > 0:
                    intent_score = intent_score / len(data['patterns'])
                    
                    if intent_score > best_score:
                        best_score = intent_score
                        best_intent = intent
            
            # Confidence threshold
            confidence = min(best_score / 5.0, 1.0) if best_score > 0 else 0.0
            
            return (best_intent, confidence) if confidence > 0.3 else (None, 0.0)
            
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return (None, 0.0)
    
    def search_knowledge_base(self, text: str) -> Optional[str]:
        """Intelligent knowledge base search"""
        try:
            processed_text = ' '.join(self.preprocess_text(text))
            
            best_match = None
            best_score = 0
            
            for topic, info in self.knowledge_base.items():
                topic_words = set(topic.split())
                text_words = set(processed_text.split())
                
                # Calculate overlap score
                overlap = len(topic_words.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(topic_words)
                    if score > best_score:
                        best_score = score
                        best_match = info
            
            if best_match and best_score > 0.3:
                if isinstance(best_match, dict):
                    return best_match.get('definition', str(best_match))
                return str(best_match)
            
            return None
            
        except Exception as e:
            logger.error(f"Knowledge base search error: {e}")
            return None
    
    def generate_contextual_response(self, user_input: str) -> Dict:
        """Generate intelligent, contextual responses"""
        try:
            # Analyze input
            sentiment_data = self.analyze_sentiment_advanced(user_input)
            entities = self.extract_entities_advanced(user_input)
            intent, intent_confidence = self.detect_intent_advanced(user_input)
            
            # Initialize response data
            response_data = {
                'text': '',
                'sentiment': sentiment_data,
                'entities': entities,
                'intent': intent,
                'intent_confidence': intent_confidence,
                'source': 'unknown'
            }
            
            # Generate response based on intent
            if intent and intent_confidence > 0.5:
                response_data['text'] = random.choice(self.intents[intent]['responses'])
                response_data['source'] = 'intent_based'
            
            # Try knowledge base if no strong intent
            elif not response_data['text']:
                kb_response = self.search_knowledge_base(user_input)
                if kb_response:
                    response_data['text'] = f"💡 {kb_response}"
                    response_data['source'] = 'knowledge_base'
            
            # Sentiment-based fallback responses
            if not response_data['text']:
                sentiment = sentiment_data['sentiment']
                
                if sentiment == 'positive':
                    fallback_responses = [
                        "That's wonderful! I love your positive energy! ✨ What else would you like to explore?",
                        "Your enthusiasm is contagious! 😊 How can I help you further?",
                        "Great to hear! What other topics interest you?",
                        "Fantastic! I'm here to help with anything else you need! 🌟"
                    ]
                elif sentiment == 'negative':
                    fallback_responses = [
                        "I understand things might be challenging. How can I help improve your day? 💙",
                        "I'm here to support you. What would make things better right now?",
                        "I hear you. Let's work together to find a solution. What do you need help with?",
                        "Your feelings are valid. How can I assist you today? 🤗"
                    ]
                else:
                    fallback_responses = [
                        "That's interesting! Could you tell me more about what you're thinking? 🤔",
                        "I'd love to help! Can you provide more context about your question?",
                        "Fascinating topic! What specific aspect would you like to explore? 🔍",
                        "I'm curious to learn more! What would you like to discuss in detail?",
                        "Great question! Could you elaborate so I can give you the best response? 💭"
                    ]
                
                response_data['text'] = random.choice(fallback_responses)
                response_data['source'] = 'sentiment_based'
            
            # Add entity context if relevant
            if entities and len(response_data['text']) < 150:
                entity_names = [ent[0] for ent in entities[:2]]  # Limit to 2 entities
                if entity_names:
                    response_data['text'] += f"\n\n🏷️ I noticed you mentioned: {', '.join(entity_names)}"
            
            # Store in conversation history
            self.conversation_history.append({
                'user_input': user_input,
                'bot_response': response_data['text'],
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'sentiment': sentiment_data,
                    'intent': intent,
                    'entities': entities
                }
            })
            
            # Update metrics
            self.response_count += 1
            if intent:
                self.intent_accuracy.append(intent_confidence)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return {
                'text': "I apologize, but I encountered an issue processing your message. Could you please try rephrasing? 🤖",
                'sentiment': {'sentiment': 'neutral', 'confidence': 0.5},
                'entities': [],
                'intent': None,
                'intent_confidence': 0.0,
                'source': 'error_handling'
            }
    
    def get_conversation_stats(self) -> Dict:
        """Get comprehensive conversation statistics"""
        try:
            if not self.conversation_history:
                return {"message": "No conversation data available"}
            
            session_duration = datetime.now() - self.session_start
            
            # Sentiment distribution
            sentiments = [msg['analysis']['sentiment']['sentiment'] 
                         for msg in self.conversation_history]
            sentiment_counts = {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            }
            
            # Intent distribution
            intents = [msg['analysis']['intent'] for msg in self.conversation_history if msg['analysis']['intent']]
            
            # Average confidence
            avg_confidence = (sum(self.intent_accuracy) / len(self.intent_accuracy)) if self.intent_accuracy else 0
            
            return {
                'session_duration': str(session_duration).split('.')[0],
                'total_exchanges': len(self.conversation_history),
                'sentiment_distribution': sentiment_counts,
                'detected_intents': len(set(intents)),
                'average_confidence': round(avg_confidence, 3),
                'most_common_intent': max(set(intents), key=intents.count) if intents else 'None'
            }
            
        except Exception as e:
            logger.error(f"Stats generation error: {e}")
            return {"error": "Unable to generate statistics"}

def display_welcome_banner():
    """Display professional welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🤖 ADVANCED AI CHATBOT WITH NLP CAPABILITIES 🤖        ║
║                                                              ║
║                   CODTECH IT SOLUTIONS                       ║
║                    Internship Task 3                        ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Features:                                                   ║
║  ✅ Advanced Intent Recognition                               ║
║  ✅ Real-time Sentiment Analysis                             ║
║  ✅ Named Entity Recognition                                 ║
║  ✅ Context-Aware Conversations                              ║
║  ✅ Comprehensive Knowledge Base                             ║
║  ✅ Conversation Analytics                                   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Commands:                                                   ║
║  💬 Just type naturally to chat                             ║
║  📊 Type 'stats' for conversation analytics                 ║
║  🔄 Type 'reset' to clear conversation history              ║
║  ❓ Type 'help' to see available features                    ║
║  👋 Type 'quit', 'exit', or 'bye' to end session           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Main application entry point"""
    try:
        # Setup NLTK
        if not NLTKSetup.setup_nltk():
            print("❌ Failed to setup NLP components. Exiting...")
            return
        
        # Display welcome banner
        display_welcome_banner()
        
        # Initialize chatbot
        print("🔄 Initializing advanced AI chatbot...")
        chatbot = AdvancedChatbot()
        print("✅ Chatbot ready! Let's start our intelligent conversation!\n")
        
        # Main conversation loop
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                # Handle empty input
                if not user_input:
                    print("🤖 Bot: Please enter a message to continue our conversation! 💬")
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🤖 Bot: Thank you for the engaging conversation! 👋")
                    print("✨ Keep learning and stay curious! Goodbye! 🌟\n")
                    break
                
                if user_input.lower() == 'stats':
                    stats = chatbot.get_conversation_stats()
                    print("\n📊 CONVERSATION ANALYTICS")
                    print("=" * 50)
                    for key, value in stats.items():
                        print(f"📈 {key.replace('_', ' ').title()}: {value}")
                    print("=" * 50 + "\n")
                    continue
                
                if user_input.lower() == 'reset':
                    chatbot.conversation_history.clear()
                    print("🤖 Bot: Conversation history cleared! Fresh start! 🔄\n")
                    continue
                
                if user_input.lower() == 'help':
                    help_text = """
🆘 CHATBOT HELP MENU
==================
🤖 I'm an advanced AI with sophisticated NLP capabilities!

💡 What I can do:
   • Have intelligent conversations
   • Analyze your emotions and sentiment  
   • Recognize named entities in your text
   • Understand your intents and respond appropriately
   • Provide information from my knowledge base
   • Remember our conversation context
   • Generate statistics about our chat

🎯 Try asking me about:
   • Technology topics (AI, ML, Python, Data Science)
   • Time and date queries
   • Jokes and entertainment
   • Motivation and inspiration
   • General questions and conversations

📊 Special commands:
   • 'stats' - View conversation analytics
   • 'reset' - Clear conversation history  
   • 'help' - Show this help menu
   • 'quit' - End our conversation

Just chat naturally - I'll understand! 🌟
"""
                    print(help_text)
                    continue
                
                # Generate response
                print("🤖 Bot: ", end="", flush=True)
                response_data = chatbot.generate_contextual_response(user_input)
                
                # Display response
                print(response_data['text'])
                
                # Display analysis (optional detailed view)
                analysis_parts = []
                
                sentiment = response_data['sentiment']
                if sentiment['sentiment'] != 'neutral':
                    analysis_parts.append(f"Sentiment: {sentiment['sentiment']} ({sentiment['confidence']})")
                
                if response_data['intent']:
                    analysis_parts.append(f"Intent: {response_data['intent']} ({response_data['intent_confidence']:.2f})")
                
                if response_data['entities']:
                    entity_count = len(response_data['entities'])
                    analysis_parts.append(f"Entities: {entity_count}")
                
                if analysis_parts:
                    print(f"   💡 Analysis: {' | '.join(analysis_parts)}")
                
                print()  # Add spacing between exchanges
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bot: Session interrupted. Thanks for chatting! 👋")
                break
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                print("🤖 Bot: I encountered an issue. Let's continue our conversation! 🔄")
        
        # Session summary
        try:
            stats = chatbot.get_conversation_stats()
            if stats.get('total_exchanges', 0) > 0:
                print("\n📊 SESSION SUMMARY")
                print("=" * 40)
                print(f"🕐 Duration: {stats.get('session_duration', 'N/A')}")
                print(f"💬 Total exchanges: {stats.get('total_exchanges', 0)}")
                print(f"🎯 Average confidence: {stats.get('average_confidence', 0)}")
                print("=" * 40)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print("❌ Critical error occurred. Please restart the application.")

if __name__ == "__main__":
    main()

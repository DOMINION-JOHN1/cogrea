import os
from groq import Groq
from elevenlabs.client import ElevenLabs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# Load environmental variables from .env file
load_dotenv()

# Retrieve API keys and environment settings
REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS")
REDIS_URL = os.getenv("REDIS_URL")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)

# Initialize Tavily search tool
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily_tool = TavilySearch(max_results=6)

# Initialize API clients
groq_client = Groq(api_key=GROQ_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
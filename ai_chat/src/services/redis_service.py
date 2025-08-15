import json
import uuid
from typing import List, Optional
from redis import Redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from config import REDIS_HOST, REDIS_PASSWORD


import json
import uuid
import logging
from typing import List, Optional
from redis import Redis, ConnectionPool
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from config import REDIS_HOST, REDIS_PASSWORD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WindowsRedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.redis =Redis(host=REDIS_HOST,port=6379,password=REDIS_PASSWORD,ssl=True)
        self.session_id = session_id
        self.key = f"chat:{session_id}"

    def add_message(self, message: BaseMessage) -> None:
        self.redis.rpush(self.key, json.dumps(message_to_dict(message)))

    @property
    def messages(self) -> List[BaseMessage]:
        _items = self.redis.lrange(self.key, 0, -1)
        return messages_from_dict([json.loads(m.decode("utf-8")) for m in _items])

    def clear(self) -> None:
        self.redis.delete(self.key)

# Redis Session Manager
class RedisSessionManager:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST,port=6379,password=REDIS_PASSWORD,ssl=True)
        self.session_ttl = 3600  # 24 hours
    def create_session(self, profile: dict) -> str:
        session_id = str(uuid.uuid4())
        self.redis.setex(
            f"session:{session_id}:profile",
            self.session_ttl,
            json.dumps(profile)
        )
        return session_id

    def get_profile(self, session_id: str) -> Optional[dict]:
        profile_data = self.redis.get(f"session:{session_id}:profile")
        return json.loads(profile_data) if profile_data else None

    def delete_session(self, session_id: str):
        keys = [f"session:{session_id}:profile", f"session:{session_id}:history"]
        self.redis.delete(*keys)

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return WindowsRedisChatMessageHistory(session_id)
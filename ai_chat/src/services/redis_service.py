import json
import uuid
import logging
from typing import List, Optional, Dict, Iterator, Any
from datetime import datetime
from redis import Redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from config import REDIS_HOST, REDIS_PASSWORD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomRedisCheckpointer(BaseCheckpointSaver):
    def __init__(self):
        super().__init__(serde=JsonPlusSerializer())
        self.redis = Redis(host=REDIS_HOST, port=6379, password=REDIS_PASSWORD, ssl=True)
    
    def get_checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        return f"checkpoint:{thread_id}:{checkpoint_id}"
    
    def get_checkpoint_ids_key(self, thread_id: str) -> str:
        return f"checkpoint_ids:{thread_id}"
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config['configurable']['thread_id']
        checkpoint_id = config['configurable'].get('checkpoint_id')
        
        if not checkpoint_id:
            # Get latest if no specific ID
            ids_key = self.get_checkpoint_ids_key(thread_id)
            checkpoint_ids = self.redis.lrange(ids_key, -1, -1)
            if not checkpoint_ids:
                return None
            checkpoint_id = checkpoint_ids[0].decode('utf-8')
        
        key = self.get_checkpoint_key(thread_id, checkpoint_id)
        data = self.redis.get(key)
        if not data:
            return None
        
        saved = json.loads(data)
        checkpoint = self.serde.loads(saved['checkpoint'])
        metadata = json.loads(saved['metadata'])  # Deserialize metadata
        parent_id = saved.get('parent_id')
        
        parent_config = None
        if parent_id:
            parent_config = {'configurable': {'thread_id': thread_id, 'checkpoint_id': parent_id}}
        
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )
    
    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata) -> RunnableConfig:
        thread_id = config['configurable']['thread_id']
        checkpoint_id = str(uuid.uuid4())
        parent_id = config['configurable'].get('checkpoint_id')
        
        key = self.get_checkpoint_key(thread_id, checkpoint_id)
        data = {
            'checkpoint': self.serde.dumps(checkpoint),
            'metadata': json.dumps(metadata),
            'parent_id': parent_id,
        }
        self.redis.set(key, json.dumps(data))
        
        ids_key = self.get_checkpoint_ids_key(thread_id)
        self.redis.rpush(ids_key, checkpoint_id)
        
        return {
            'configurable': {
                'thread_id': thread_id,
                'checkpoint_id': checkpoint_id,
            }
        }
    
    def list(self, config: Optional[RunnableConfig], *, filter: Optional[Dict[str, Any]] = None, before: Optional[RunnableConfig] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        thread_id = config['configurable']['thread_id'] if config else None
        if not thread_id:
            return
        ids_key = self.get_checkpoint_ids_key(thread_id)
        checkpoint_ids = self.redis.lrange(ids_key, 0, -1)
        for cid_bytes in reversed(checkpoint_ids):  # Latest first
            cid = cid_bytes.decode('utf-8')
            cp_config = {'configurable': {'thread_id': thread_id, 'checkpoint_id': cid}}
            tuple_ = self.get_tuple(cp_config)
            if tuple_:
                yield tuple_


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
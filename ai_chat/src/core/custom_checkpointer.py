import json
import uuid
from typing import Dict, Optional, Iterator,Any
from datetime import datetime
from redis import Redis
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, empty_checkpoint,CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
from config import REDIS_HOST, REDIS_PASSWORD


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


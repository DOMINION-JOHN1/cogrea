import json
import uuid
from typing import Dict, Optional, Iterator, Any, Sequence
from datetime import datetime
from redis import Redis
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, empty_checkpoint, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
from config import REDIS_HOST, REDIS_PASSWORD
import base64
from json import JSONEncoder

# A custom JSON encoder that handles bytes objects
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            # Encode bytes to a Base64 string
            return base64.b64encode(obj).decode('utf-8')
        # Let the default encoder handle other types
        return JSONEncoder.default(self, obj)

class CustomRedisCheckpointer(BaseCheckpointSaver):
    def __init__(self):
        super().__init__(serde=JsonPlusSerializer())
        self.redis = Redis(host=REDIS_HOST, port=6379, password=REDIS_PASSWORD, ssl=True)
    
    def get_checkpoint_key(self, thread_id: str, checkpoint_ts: str) -> str:
        return f"checkpoint:{thread_id}:{checkpoint_ts}"
    
    def get_checkpoint_ids_key(self, thread_id: str) -> str:
        return f"checkpoint_ids:{thread_id}"
    
    def get_writes_key(self, thread_id: str, checkpoint_ts: str) -> str:
        return f"writes:{thread_id}:{checkpoint_ts}"
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config['configurable']['thread_id']
        checkpoint_ts = config['configurable'].get('thread_ts')
        
        print(f"CHECKPOINTER DEBUG: Getting tuple for thread {thread_id}, ts {checkpoint_ts}")
        
        if not checkpoint_ts:
            # Get latest if no specific timestamp
            ids_key = self.get_checkpoint_ids_key(thread_id)
            checkpoint_timestamps = self.redis.lrange(ids_key, -1, -1)
            print(f"CHECKPOINTER DEBUG: Found {len(checkpoint_timestamps)} timestamps for thread")
            if not checkpoint_timestamps:
                print("CHECKPOINTER DEBUG: No timestamps found, returning None")
                return None
            checkpoint_ts = checkpoint_timestamps[0].decode('utf-8')
            print(f"CHECKPOINTER DEBUG: Using latest timestamp: {checkpoint_ts}")
        
        key = self.get_checkpoint_key(thread_id, checkpoint_ts)
        data = self.redis.get(key)
        if not data:
            print(f"CHECKPOINTER DEBUG: No data found for key: {key}")
            return None
        
        try:
            saved = json.loads(data)
            
            # Debug logging to see what we're trying to deserialize
            checkpoint_data = saved.get('checkpoint')
            if not checkpoint_data:
                print(f"CHECKPOINTER DEBUG: Empty checkpoint data for key: {key}")
                return None
            
            print(f"CHECKPOINTER DEBUG: Checkpoint data type: {type(checkpoint_data)}")
            
            # Handle both string and bytes checkpoint data
            if isinstance(checkpoint_data, str):
                # If it's base64 encoded bytes, decode it first
                try:
                    checkpoint_bytes = base64.b64decode(checkpoint_data)
                    checkpoint = self.serde.loads(checkpoint_bytes)
                except:
                    # If base64 decode fails, treat it as direct string
                    checkpoint = self.serde.loads(checkpoint_data)
            else:
                checkpoint = self.serde.loads(checkpoint_data)
            
            print(f"CHECKPOINTER DEBUG: Successfully loaded checkpoint with keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'not a dict'}")
            
            metadata = saved.get('metadata', {})
            parent_ts = saved.get('parent_ts')
            
            parent_config = None
            if parent_ts:
                parent_config = {'configurable': {'thread_id': thread_id, 'thread_ts': parent_ts}}
            
            result = CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )
            
            print(f"CHECKPOINTER DEBUG: Returning checkpoint tuple")
            return result
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"CHECKPOINTER DEBUG: Error parsing checkpoint data for key {key}: {e}")
            print(f"CHECKPOINTER DEBUG: Raw data: {data[:100] if data else 'None'}...")
            # If we can't parse the checkpoint, return None to create a fresh one
            return None
    
    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, parent_config: Optional[RunnableConfig] = None) -> RunnableConfig:
        """
        Save a checkpoint with the correct method signature that matches the current LangGraph interface.
        
        Args:
            config: The runnable config
            checkpoint: The checkpoint data to save
            metadata: The checkpoint metadata
            parent_config: Optional parent config for nested checkpoints
        """
        thread_id = config['configurable']['thread_id']
        # Use the checkpoint timestamp as the unique identifier
        checkpoint_ts = checkpoint['ts']
        
        print(f"CHECKPOINTER DEBUG: Storing checkpoint for thread {thread_id}, ts {checkpoint_ts}")
        
        # Safely extract parent timestamp
        parent_ts = None
        if parent_config and isinstance(parent_config, dict):
            if 'configurable' in parent_config:
                parent_ts = parent_config['configurable'].get('thread_ts')
            elif 'thread_ts' in parent_config:
                parent_ts = parent_config.get('thread_ts')
        
        key = self.get_checkpoint_key(thread_id, checkpoint_ts)
        
        # Serialize the checkpoint data
        try:
            serialized_checkpoint = self.serde.dumps(checkpoint)
            print(f"CHECKPOINTER DEBUG: Serialized checkpoint type: {type(serialized_checkpoint)}")
            print(f"CHECKPOINTER DEBUG: Serialized checkpoint length: {len(serialized_checkpoint)}")
            
            # Handle different serialization formats
            if isinstance(serialized_checkpoint, bytes):
                # Convert bytes to base64 string for JSON storage
                checkpoint_data = base64.b64encode(serialized_checkpoint).decode('utf-8')
            else:
                # Already a string
                checkpoint_data = serialized_checkpoint
            
            data = {
                'checkpoint': checkpoint_data,
                'metadata': metadata,
                'parent_ts': parent_ts,
            }
            
            # Store in Redis using standard json.dumps (no CustomJSONEncoder needed now)
            self.redis.set(key, json.dumps(data))
            print(f"CHECKPOINTER DEBUG: Successfully stored checkpoint with key: {key}")
            
            # Store the timestamp in the ordered list
            ids_key = self.get_checkpoint_ids_key(thread_id)
            self.redis.rpush(ids_key, checkpoint_ts)
            
            return {
                'configurable': {
                    'thread_id': thread_id,
                    'thread_ts': checkpoint_ts,
                }
            }
            
        except Exception as e:
            print(f"CHECKPOINTER DEBUG: Error storing checkpoint: {e}")
            print(f"CHECKPOINTER DEBUG: Checkpoint data: {checkpoint}")
            raise
    
    def put_writes(self, config: RunnableConfig, writes: Sequence[tuple], task_id: str) -> None:
        """
        Store intermediate writes linked to a checkpoint (i.e. pending writes).
        
        Args:
            config: The runnable config
            writes: Sequence of (channel, value) writes to store
            task_id: Unique identifier for this set of writes
        """
        thread_id = config['configurable']['thread_id']
        checkpoint_ts = config['configurable'].get('thread_ts', '')
        
        # Store writes with task_id as part of the key
        writes_key = f"{self.get_writes_key(thread_id, checkpoint_ts)}:{task_id}"
        
        # Serialize writes
        writes_data = []
        for channel, value in writes:
            serialized_value = None
            if value is not None:
                serialized_value = self.serde.dumps(value)
                # Handle bytes in writes too
                if isinstance(serialized_value, bytes):
                    serialized_value = base64.b64encode(serialized_value).decode('utf-8')
            
            writes_data.append({
                'channel': channel,
                'value': serialized_value
            })
        
        # Use standard json.dumps since we've handled bytes conversion above
        self.redis.set(writes_key, json.dumps(writes_data))
        
        # Also maintain a list of task_ids for this checkpoint
        task_ids_key = f"{self.get_writes_key(thread_id, checkpoint_ts)}:task_ids"
        self.redis.rpush(task_ids_key, task_id)
    
    def list(self, config: Optional[RunnableConfig], *, filter: Optional[Dict[str, Any]] = None, before: Optional[RunnableConfig] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        thread_id = config['configurable']['thread_id'] if config else None
        if not thread_id:
            return
        
        ids_key = self.get_checkpoint_ids_key(thread_id)
        checkpoint_timestamps = self.redis.lrange(ids_key, 0, -1)
        
        count = 0
        for ts_bytes in reversed(checkpoint_timestamps):  # Latest first
            if limit is not None and count >= limit:
                break
                
            ts = ts_bytes.decode('utf-8')
            cp_config = {'configurable': {'thread_id': thread_id, 'thread_ts': ts}}
            tuple_ = self.get_tuple(cp_config)
            if tuple_:
                yield tuple_
                count += 1
"""
Ollama客戶端
提供與Ollama服務的統一接口
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import httpx
from datetime import datetime

from config.settings import get_settings


class OllamaClient:
    """Ollama客戶端"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama.host.rstrip('/')
        self.timeout = self.settings.ollama.timeout
        self.max_retries = self.settings.ollama.max_retries
        
        # 創建HTTP客戶端
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """發送HTTP請求"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == self.max_retries:
                    raise ConnectionError(f"Failed to connect to Ollama after {self.max_retries + 1} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # 指數退避
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        try:
            response = await self._make_request("GET", "/api/tags")
            return response.get("models", [])
        except Exception as e:
            raise ValueError(f"Failed to list models: {e}")
    
    async def check_model_exists(self, model_name: str) -> bool:
        """檢查模型是否存在"""
        try:
            models = await self.list_models()
            return any(model["name"] == model_name for model in models)
        except:
            return False
    
    async def pull_model(self, model_name: str) -> AsyncGenerator[str, None]:
        """下載模型"""
        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name}
        
        try:
            async with self._client.stream(
                "POST", url, json=payload, timeout=None
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data.get("status", "")
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise ConnectionError(f"Failed to pull model {model_name}: {e}")
    
    async def generate_text(self, 
                          prompt: str, 
                          model: Optional[str] = None,
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          stream: bool = False,
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """生成文本"""
        model = model or self.settings.ollama.chat_model
        temperature = temperature or self.settings.ollama.chat_temperature
        max_tokens = max_tokens or self.settings.ollama.chat_max_tokens
        
        # 檢查模型是否存在
        if not await self.check_model_exists(model):
            raise ValueError(f"Model {model} not found. Please pull it first.")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        if stream:
            return self._generate_stream(payload)
        else:
            response = await self._make_request("POST", "/api/generate", json=payload)
            return response.get("response", "")
    
    async def _generate_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        url = f"{self.base_url}/api/generate"
        
        try:
            async with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise ConnectionError(f"Failed to generate text: {e}")
    
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   model: Optional[str] = None,
                   temperature: Optional[float] = None,
                   stream: bool = False,
                   **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """聊天對話"""
        model = model or self.settings.ollama.chat_model
        temperature = temperature or self.settings.ollama.chat_temperature
        
        if not await self.check_model_exists(model):
            raise ValueError(f"Model {model} not found. Please pull it first.")
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                **kwargs
            }
        }
        
        if stream:
            return self._chat_stream(payload)
        else:
            response = await self._make_request("POST", "/api/chat", json=payload)
            return response.get("message", {}).get("content", "")
    
    async def _chat_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """流式聊天"""
        url = f"{self.base_url}/api/chat"
        
        try:
            async with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise ConnectionError(f"Failed to chat: {e}")
    
    async def generate_embeddings(self, 
                                text: Union[str, List[str]], 
                                model: Optional[str] = None) -> Union[List[float], List[List[float]]]:
        """生成嵌入向量"""
        model = model or self.settings.ollama.embedding_model
        
        if not await self.check_model_exists(model):
            raise ValueError(f"Embedding model {model} not found. Please pull it first.")
        
        # 處理批量輸入
        if isinstance(text, str):
            texts = [text]
            return_single = True
        else:
            texts = text
            return_single = False
        
        # 批量處理嵌入
        batch_size = self.settings.ollama.embedding_batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for txt in batch:
                payload = {
                    "model": model,
                    "prompt": txt
                }
                
                try:
                    response = await self._make_request("POST", "/api/embeddings", json=payload)
                    embedding = response.get("embedding", [])
                    if not embedding:
                        raise ValueError("Empty embedding returned")
                    batch_embeddings.append(embedding)
                except Exception as e:
                    raise ValueError(f"Failed to generate embedding for text: {e}")
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings[0] if return_single else all_embeddings
    
    async def generate_code(self, 
                          prompt: str, 
                          language: str = "python",
                          model: Optional[str] = None,
                          temperature: Optional[float] = None) -> str:
        """生成代碼"""
        model = model or self.settings.ollama.code_model
        temperature = temperature or self.settings.ollama.code_temperature
        
        # 構建代碼生成提示
        code_prompt = f"""Generate {language} code for the following request:

{prompt}

Please provide only the code without explanation unless specifically requested.
Make sure the code is well-commented and follows best practices.

{language.upper()} CODE:
"""
        
        return await self.generate_text(
            prompt=code_prompt,
            model=model,
            temperature=temperature
        )
    
    async def health_check(self) -> bool:
        """健康檢查"""
        try:
            response = await self._make_request("GET", "/api/tags")
            return True
        except:
            return False
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """獲取模型信息"""
        try:
            payload = {"name": model_name}
            response = await self._make_request("POST", "/api/show", json=payload)
            return response
        except Exception as e:
            raise ValueError(f"Failed to get model info for {model_name}: {e}")


class OllamaModelManager:
    """Ollama模型管理器"""
    
    def __init__(self):
        self.client = OllamaClient()
        self.settings = get_settings()
    
    async def __aenter__(self):
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def ensure_models_ready(self) -> Dict[str, bool]:
        """確保所需模型已準備就緒"""
        required_models = [
            self.settings.ollama.chat_model,
            self.settings.ollama.embedding_model,
            self.settings.ollama.code_model
        ]
        
        model_status = {}
        
        for model in required_models:
            exists = await self.client.check_model_exists(model)
            model_status[model] = exists
            
            if not exists:
                print(f"Model {model} not found. Starting download...")
                async for status in self.client.pull_model(model):
                    print(f"Downloading {model}: {status}")
                model_status[model] = True
        
        return model_status
    
    async def get_available_models(self) -> List[str]:
        """獲取可用模型列表"""
        models = await self.client.list_models()
        return [model["name"] for model in models]
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """驗證配置"""
        try:
            # 檢查連接
            health = await self.client.health_check()
            
            # 檢查模型
            model_status = await self.ensure_models_ready()
            
            return {
                "connection": health,
                "models": model_status,
                "status": "ready" if health and all(model_status.values()) else "not_ready"
            }
        except Exception as e:
            return {
                "connection": False,
                "models": {},
                "status": "error",
                "error": str(e)
            }


# 全局客戶端實例
_ollama_client = None


async def get_ollama_client() -> OllamaClient:
    """獲取Ollama客戶端實例"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


async def cleanup_ollama_client():
    """清理客戶端實例"""
    global _ollama_client
    if _ollama_client is not None:
        await _ollama_client.__aexit__(None, None, None)
        _ollama_client = None


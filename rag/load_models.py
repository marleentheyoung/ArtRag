import yaml
import torch
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import Tuple, Any


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(fmt='[%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class ModelLoader:
    def __init__(self, config_path: str = "rag/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("ModelLoader")
        
        self.device = self._get_device()
        self.embedding_device = self._get_embedding_device()
        self.generator_device = self._get_generator_device()
        self.logger.info(f"Embedding device: {self.embedding_device}")
        self.logger.info(f"Generator device: {self.generator_device}")
    
    def _detect_best_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_device(self) -> str:
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            return self._detect_best_device()
        else:
            if device_config == 'cuda' and not torch.cuda.is_available():
                return self._detect_best_device()
            elif device_config == 'mps' and not torch.backends.mps.is_available():
                return self._detect_best_device()
            
            return device_config
    
    def _get_embedding_device(self) -> str:
        embedding_device = self.config.get('embedding_device', self.device)
        if embedding_device == 'auto':
            return self._detect_best_device()
        return embedding_device
    
    def _get_generator_device(self) -> str:
        generator_device = self.config.get('generator_device', self.device)
        
        if generator_device == 'auto':
            return self._detect_best_device()
        
        return generator_device
    
    def _get_torch_dtype(self) -> torch.dtype:
        dtype_config = self.config.get('torch_dtype', 'auto')
        
        if dtype_config == 'auto':
            if self.generator_device in ['cuda', 'mps']:
                return torch.float16
            else:
                return torch.float32
        elif dtype_config == 'float16':
            return torch.float16
        elif dtype_config == 'float32':
            return torch.float32
        else:
            self.logger.warning(f"Unknown torch_dtype: {dtype_config}, using float32")
            return torch.float32
    
    def load_embedding_model(self) -> Tuple[Any, Any]:
        self.logger.info("Loading embedding model...")
        model_path = self.config.get('embedding_model_name', "Qwen/Qwen3-Embedding-0.6B")
        
        model = SentenceTransformer(
            model_path,
            device=self.embedding_device,
            trust_remote_code=True
        )
        
        self.logger.info(f"Embedding model loaded on {self.embedding_device}")
        return None, model
    
    def load_generator_model(self) -> Tuple[Any, Any]:
        self.logger.info("Loading generator model...")
        
        model_path = self.config.get('generator_model_name', "Qwen/Qwen2.5-1.5B-Instruct")
        torch_dtype = self._get_torch_dtype()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model_kwargs = {'torch_dtype': torch_dtype, 'trust_remote_code': True}
        
        if self.generator_device == 'cuda' and torch.cuda.device_count() > 1:
            model_kwargs['device_map'] = 'auto'

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if 'device_map' not in model_kwargs:
            model = model.to(self.generator_device)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.logger.info(f"Generator model loaded on {self.generator_device} with dtype {torch_dtype}")
        return tokenizer, model
    
    def get_device(self) -> str:
        return self.device
    
    def get_embedding_device(self) -> str:
        return self.embedding_device
    
    def get_generator_device(self) -> str:
        return self.generator_device
        
        return info

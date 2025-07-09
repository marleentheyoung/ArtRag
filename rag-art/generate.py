import yaml
from typing import List, Dict, Any
import torch
from load_models import setup_logger


class ResponseGenerator:
    def __init__(self, config_path: str = "config.yaml", tokenizer=None, model=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("Generator")
        self.tokenizer = tokenizer
        self.model = model
        self.device = getattr(model, 'device', 'cpu')
    
    def _format_document(self, doc: Dict[str, Any], index: int) -> str:
        doc_type = doc.get('doc_type', 'document')
        
        if doc_type == 'artwork':
            return f"[Artwork {index}] {doc['text']}"
        elif doc_type == 'artist':
            return f"[Artist {index}] {doc['text']}"
        else:
            return f"[{index}] {doc['text']}"
    
    def _create_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            user_content = f"Question: {query}"
        else:
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                formatted_doc = self._format_document(doc, i)
                context_parts.append(formatted_doc)
            
            context = "\n\n".join(context_parts)
            
            type_counts = {}
            for doc in retrieved_docs:
                doc_type = doc.get('doc_type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            type_info = []
            if type_counts.get('artwork', 0) > 0:
                type_info.append(f"{type_counts['artwork']} artwork(s)")
            if type_counts.get('artist', 0) > 0:
                type_info.append(f"{type_counts['artist']} artist(s)")
            
            context_description = f"information from {', '.join(type_info)}" if type_info else "the provided information"
            
            user_content = f"""Answer the question using {context_description}. Do not add information not present in the context.

Context:
{context}

Question: {query}"""
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that provides accurate information about artworks and artists."},
            {"role": "user", "content": user_content}
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        prompt = self._create_prompt(query, retrieved_docs)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=2048
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_generation_tokens', 500),
                do_sample=self.config.get('do_sample', True),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.config.get('repetition_penalty', 1.2)
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def generate_with_citations(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        response = self.generate(query, retrieved_docs)
        
        type_counts = {}
        for doc in retrieved_docs:
            doc_type = doc.get('doc_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        citation_info = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_type = doc.get('doc_type', 'unknown')
            score = doc.get('score', 0.0)
            cluster_id = doc.get('cluster_id', None)
            
            citation = {
                'index': i,
                'doc_type': doc_type,
                'score': score,
                'text_preview': doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
            }
            
            if cluster_id is not None:
                citation['cluster_id'] = cluster_id
            
            citation_info.append(citation)
        
        return {
            'response': response,
            'citations': citation_info,
            'source_summary': {
                'total_sources': len(retrieved_docs),
                'type_breakdown': type_counts
            }
        }
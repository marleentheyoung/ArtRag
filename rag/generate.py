import yaml
from typing import List, Dict, Any
import torch
from rag.load_models import setup_logger


class ResponseGenerator:
    def __init__(self, config_path: str = "rag/config.yaml", tokenizer=None, model=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("Generator")
        self.tokenizer = tokenizer
        self.model = model
        self.device = getattr(model, 'device', 'cpu')
    
    def _format_document(self, doc: Dict[str, Any], index: int) -> str:
        doc_type = doc.get('doc_type', 'document')
        
        if doc_type == 'artwork':
            picture_id = doc.get('picture_id')
            if picture_id:
                return f"[Artwork {index} - ID: {picture_id}] {doc['text']}"
            else:
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
            
            user_content = f"""Answer the question using {context_description}. Be precise and extract specific numbers, dates, or facts when asked. Do not add information not present in the context.

Context:
{context}

Question: {query}"""
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that provides accurate information about artworks and artists. When asked for specific numbers or facts, extract them directly from the provided context."},
            {"role": "user", "content": user_content}
        ]
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        prompt = self._create_prompt(query, retrieved_docs)
        # self.logger.debug(f"Generated prompt length: {len(prompt)} characters")
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=4096 
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generation_config = {
            'max_new_tokens': self.config.get('max_generation_tokens', 250),
            'do_sample': self.config.get('do_sample', True),
            'temperature': self.config.get('temperature', 0.3),
            'top_p': self.config.get('top_p', 0.9),
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': self.config.get('repetition_penalty', 1.2),
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = self._clean_response(response)
        return response
    
    def _clean_response(self, response: str) -> str:
        response = response.strip()
        
        prefixes_to_remove = [
            "assistant",
            "<|im_start|>assistant",
            "Assistant:",
            "ASSISTANT:",
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        system_patterns = [
            "You are Qwen, created by Alibaba Cloud.",
            "user\nAnswer the question",
        ]
        
        for pattern in system_patterns:
            if pattern in response:
                parts = response.split(pattern)
                if len(parts) > 1:
                    # after the system message
                    response = parts[-1].strip()
        
        if "Question:" in response:
            question_parts = response.split("Question:")
            if len(question_parts) > 2:  # More than one "Question:" found
                # Take everything before the last "Question:"
                response = "Question:".join(question_parts[:-1]).strip()
            elif len(question_parts) == 2:
                second_part = question_parts[1].strip()
                if len(second_part) < 50:  # rep
                    response = question_parts[0].strip()
        
        artifacts_to_remove = [
            "For **",
            "For *",
            "**For",
            "*For",
        ]
        
        for artifact in artifacts_to_remove:
            if response.endswith(artifact):
                response = response.rsplit(artifact, 1)[0].strip()
        
        if len(response) > 1200:
            sentences = response.split(". ")
            if len(sentences) > 8:  # only truncate if there are many sentences
                response = ". ".join(sentences[:6]) + "."
        
        if response.count(";") > 15:
            semicolon_parts = response.split(";")
            meaningful_parts = []
            for part in semicolon_parts[:5]:
                part = part.strip()
                if len(part) > 10:
                    meaningful_parts.append(part)
            if meaningful_parts:
                response = ". ".join(meaningful_parts) + "."
        
        # Final cleanup
        response = response.strip()
        
        # Ensure the response doesn't end abruptly in the middle of a word
        if response and not response[-1] in '.!?':
            for ending in ['.', '!', '?']:
                last_ending = response.rfind(ending)
                if last_ending > len(response) * 0.8:
                    response = response[:last_ending + 1]
                    break
        
        return response
    
    def generate_with_citations(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        response = self.generate(query, retrieved_docs)
        
        type_counts = {}
        artwork_count = 0
        for doc in retrieved_docs:
            doc_type = doc.get('doc_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            if doc_type == 'artwork':
                artwork_count += 1
        
        citation_info = []
        picture_ids = []
        
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
            
            if doc_type == 'artwork' and 'picture_id' in doc:
                citation['picture_id'] = doc['picture_id']
                picture_ids.append(doc['picture_id'])
            
            if cluster_id is not None:
                citation['cluster_id'] = cluster_id
            
            citation_info.append(citation)
        
        result = {
            'response': response,
            'citations': citation_info,
            'source_summary': {
                'total_sources': len(retrieved_docs),
                'type_breakdown': type_counts
            }
        }
        
        if picture_ids:
            result['picture_ids'] = picture_ids
            result['source_summary']['artwork_picture_ids'] = picture_ids
        
        return result

"""
evaluation script for Art RAG system
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
import statistics
import warnings
from collections import defaultdict
from difflib import SequenceMatcher
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For text metrics
try:
    from rouge import Rouge
    from bert_score import BERTScorer
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import word_tokenize
except ImportError:
    logger.warning("Some packages not installed. Installing required packages...")
    import subprocess
    import sys
    
    packages = ['rouge-score', 'nltk', 'bert-score']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    from rouge import Rouge
    from bert_score import BERTScorer
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import word_tokenize


class RobustAnswerExtractor:
    """Robust answer extraction with better logic and validation"""
    
    def __init__(self):
        self.number_patterns = [
            r'\*\*(\d+)\*\*',  # Bold numbers
            r'(\d+)\s+unique',  # "X unique"
            r'(\d+)\s+different',  # "X different"
            r'(\d+)\s+exhibitions?',  # "X exhibitions"
            r'(\d+)\s+countries?',  # "X countries"
            r'(\d+)\s+artists?',  # "X artists"
            r'(\d+)\s+cities?',  # "X cities"
            r'\b(\d+)\b'  # Any number as fallback
        ]
        
        self.yes_no_patterns = [
            r'\byes\b',
            r'\bno\b'
        ]
        
        self.name_patterns = [
            r'\*\*([^*]+)\*\*',  # Bold text
            r'"([^"]+)"',  # Quoted text
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # Proper names
        ]
    
    def clean_response(self, response: str) -> str:
        """Clean and validate response text"""
        if not response:
            return ""
        
        # Remove truncated parts (common issue)
        if "1Human:" in response:
            response = response.split("1Human:")[0].strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        return response.strip()
    
    def extract_numerical_answer(self, response: str, query: str) -> Optional[str]:
        """Extract numerical answers with context awareness"""
        response = self.clean_response(response)
        
        # Try patterns in order of specificity
        for pattern in self.number_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # For numerical questions, prefer the most prominent number
                numbers = [int(m) for m in matches if m.isdigit()]
                if numbers:
                    # If there are multiple numbers, use context to choose
                    if len(numbers) == 1:
                        return str(numbers[0])
                    else:
                        # Choose based on query context
                        return str(self._choose_best_number(numbers, query, response))
        
        return None
    
    def _choose_best_number(self, numbers: List[int], query: str, response: str) -> int:
        """Choose the most appropriate number based on context"""
        # If query asks for exhibitions, prefer larger numbers
        if 'exhibition' in query.lower():
            return max(numbers)
        
        # If query asks for countries, prefer smaller numbers (more realistic)
        if 'countr' in query.lower():
            return min([n for n in numbers if n <= 20], default=numbers[0])
        
        # If query asks for artists, prefer moderate numbers
        if 'artist' in query.lower():
            return min([n for n in numbers if 5 <= n <= 100], default=numbers[0])
        
        # Default: return first number
        return numbers[0]
    
    def extract_yes_no_answer(self, response: str) -> Optional[str]:
        """Extract yes/no answers"""
        response = self.clean_response(response).lower()
        
        # Look for clear yes/no at the beginning
        if response.startswith('yes'):
            return "Yes"
        if response.startswith('no'):
            return "No"
        
        # Count yes/no occurrences
        yes_count = len(re.findall(r'\byes\b', response))
        no_count = len(re.findall(r'\bno\b', response))
        
        if yes_count > no_count:
            return "Yes"
        elif no_count > yes_count:
            return "No"
        
        return None
    
    def extract_name_answer(self, response: str) -> Optional[str]:
        """Extract names or specific text answers"""
        response = self.clean_response(response)
        
        # Try bold text first
        bold_matches = re.findall(r'\*\*([^*]+)\*\*', response)
        if bold_matches:
            return bold_matches[0].strip()
        
        # Try quoted text
        quoted_matches = re.findall(r'"([^"]+)"', response)
        if quoted_matches:
            return quoted_matches[0].strip()
        
        # Try proper names
        name_matches = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', response)
        if name_matches:
            return name_matches[0].strip()
        
        return None
    
    def extract_answer(self, response: str, query: str, ground_truth: str) -> str:
        """Extract answer based on query type and ground truth pattern"""
        response = self.clean_response(response)
        
        if not response:
            return ""
        
        # Determine answer type from ground truth
        if ground_truth.strip().isdigit():
            # Numerical answer expected
            answer = self.extract_numerical_answer(response, query)
            if answer:
                return answer
        
        elif ground_truth.lower() in ['yes', 'no']:
            # Yes/no answer expected
            answer = self.extract_yes_no_answer(response)
            if answer:
                return answer
        
        else:
            # Text answer expected
            answer = self.extract_name_answer(response)
            if answer:
                return answer
        
        # Fallback: return first meaningful sentence
        sentences = response.split('.')
        if sentences:
            return sentences[0].strip()[:100]
        
        return response[:100]


class RobustRetrievalEvaluator:
    """Improved retrieval evaluation with better answer support detection"""
    
    def __init__(self):
        self.k_values = [1, 3, 5, 10]
        self.answer_extractor = RobustAnswerExtractor()
    
    def check_answer_support(self, ground_truth: str, retrieved_docs: List[Dict]) -> bool:
        """Check if retrieved documents contain information to support the correct answer"""
        if not retrieved_docs or not ground_truth:
            return False
        
        gt_clean = ground_truth.strip().lower()
        
        # For numerical answers
        if gt_clean.isdigit():
            target_number = gt_clean
            for doc in retrieved_docs:
                doc_text = doc.get('text', '').lower()
                if target_number in doc_text:
                    return True
        
        # For text answers
        else:
            # Split ground truth into meaningful parts
            gt_parts = [part.strip() for part in gt_clean.split() if len(part.strip()) > 2]
            
            for doc in retrieved_docs:
                doc_text = doc.get('text', '').lower()
                
                # Check if any significant part of ground truth appears in document
                matches = sum(1 for part in gt_parts if part in doc_text)
                if matches >= len(gt_parts) * 0.5:  # At least 50% of parts match
                    return True
        
        return False
    
    def evaluate_query(self, query: str, retrieved_docs: List[Dict], ground_truth: str) -> Dict[str, float]:
        """Evaluate a single query's retrieval"""
        metrics = {}
        
        # Calculate metrics based on answer support
        for k in self.k_values:
            top_k_docs = retrieved_docs[:k]
            supports_in_top_k = self.check_answer_support(ground_truth, top_k_docs)
            
            # Hit rate: Does top-k contain answer-supporting documents?
            metrics[f'hit_rate@{k}'] = 1.0 if supports_in_top_k else 0.0
            
            # Precision: What fraction of top-k documents support the answer?
            if top_k_docs:
                supporting_docs = sum(1 for doc in top_k_docs if self.check_answer_support(ground_truth, [doc]))
                metrics[f'precision@{k}'] = supporting_docs / len(top_k_docs)
            else:
                metrics[f'precision@{k}'] = 0.0
        
        # MRR: Position of first answer-supporting document
        for i, doc in enumerate(retrieved_docs):
            if self.check_answer_support(ground_truth, [doc]):
                metrics['mrr'] = 1.0 / (i + 1)
                break
        else:
            metrics['mrr'] = 0.0
        
        return metrics


class RobustGenerationEvaluator:
    """Improved generation evaluation with better answer extraction"""
    
    def __init__(self):
        self.rouge = Rouge()
        self.bertscore = BERTScorer(model_type="bert-base-uncased")
        self.answer_extractor = RobustAnswerExtractor()
    
    def exact_match_score(self, ground_truth: str, response: str, query: str) -> float:
        """Calculate exact match score with improved extraction"""
        gt_clean = ground_truth.strip()
        pred_answer = self.answer_extractor.extract_answer(response, query, gt_clean)
        
        if not pred_answer:
            return 0.0
        
        return 1.0 if gt_clean.lower() == pred_answer.lower().strip() else 0.0
    
    def numerical_accuracy(self, ground_truth: str, response: str, query: str) -> float:
        """Calculate numerical accuracy with improved extraction"""
        gt_clean = ground_truth.strip()
        
        if not gt_clean.isdigit():
            return 0.0
        
        pred_answer = self.answer_extractor.extract_numerical_answer(response, query)
        
        if not pred_answer or not pred_answer.isdigit():
            return 0.0
        
        try:
            gt_num = int(gt_clean)
            pred_num = int(pred_answer)
            
            if gt_num == pred_num:
                return 1.0
            elif gt_num == 0:
                return 0.0
            else:
                # Calculate relative error
                error = abs(gt_num - pred_num) / abs(gt_num)
                return max(0.0, 1.0 - error)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def contains_answer_score(self, ground_truth: str, response: str, query: str) -> float:
        """Check if response contains the ground truth answer"""
        response_clean = self.answer_extractor.clean_response(response)
        gt_clean = ground_truth.strip().lower()
        response_lower = response_clean.lower()
        
        if gt_clean in response_lower:
            return 1.0
        
        # For numerical answers
        if gt_clean.isdigit():
            extracted = self.answer_extractor.extract_numerical_answer(response, query)
            if extracted and extracted == gt_clean:
                return 1.0
        
        # For text answers - check partial matches
        gt_words = set(gt_clean.split())
        response_words = set(response_lower.split())
        
        if not gt_words:
            return 0.0
        
        overlap = len(gt_words.intersection(response_words))
        return overlap / len(gt_words)
    
    def fuzzy_match_score(self, ground_truth: str, response: str, query: str) -> float:
        """Calculate fuzzy string matching score"""
        gt_clean = ground_truth.strip().lower()
        pred_answer = self.answer_extractor.extract_answer(response, query, ground_truth)
        
        if not pred_answer:
            return 0.0
        
        pred_clean = pred_answer.lower().strip()
        
        return SequenceMatcher(None, gt_clean, pred_clean).ratio()
    
    def calculate_semantic_similarity(self, ground_truth: str, response: str) -> float:
        """Calculate semantic similarity using BERTScore"""
        try:
            response_clean = self.answer_extractor.clean_response(response)
            if not response_clean or not ground_truth:
                return 0.0
            
            P, R, F1 = self.bertscore.score([response_clean], [ground_truth])
            return F1.item()
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return 0.0
    
    def evaluate_generation(self, ground_truth: str, response: str, query: str) -> Dict[str, float]:
        """Evaluate generation quality with improved metrics"""
        metrics = {}
        
        # Core metrics
        metrics['exact_match'] = self.exact_match_score(ground_truth, response, query)
        metrics['numerical_accuracy'] = self.numerical_accuracy(ground_truth, response, query)
        metrics['fuzzy_match'] = self.fuzzy_match_score(ground_truth, response, query)
        metrics['contains_answer'] = self.contains_answer_score(ground_truth, response, query)
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(ground_truth, response)
        
        # ROUGE metrics
        try:
            response_clean = self.answer_extractor.clean_response(response)
            if ground_truth.strip() and response_clean:
                rouge_scores = self.rouge.get_scores(response_clean, ground_truth)[0]
                metrics['rouge_1_f'] = rouge_scores['rouge-1']['f']
                metrics['rouge_l_f'] = rouge_scores['rouge-l']['f']
            else:
                metrics['rouge_1_f'] = 0.0
                metrics['rouge_l_f'] = 0.0
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            metrics['rouge_1_f'] = 0.0
            metrics['rouge_l_f'] = 0.0
        
        return metrics


class RobustRAGEvaluator:
    """Main evaluator with robust logic and validation"""
    
    def __init__(self, query_logs_path: str, ground_truth_path: str):
        self.query_logs_path = query_logs_path
        self.ground_truth_path = ground_truth_path
        self.retrieval_evaluator = RobustRetrievalEvaluator()
        self.generation_evaluator = RobustGenerationEvaluator()
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load query logs and ground truth with validation"""
        logger.info("Loading data...")
        
        # Load query logs
        try:
            with open(self.query_logs_path, 'r', encoding='utf-8') as f:
                self.query_logs = json.load(f)
            logger.info(f"Loaded {len(self.query_logs)} query logs")
        except Exception as e:
            logger.error(f"Failed to load query logs: {e}")
            raise
        
        # Load ground truth
        try:
            self.ground_truth = pd.read_csv(self.ground_truth_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.ground_truth = pd.read_csv(self.ground_truth_path, encoding='latin-1')
                logger.info("Loaded CSV with latin-1 encoding")
            except UnicodeDecodeError:
                self.ground_truth = pd.read_csv(self.ground_truth_path, encoding='utf-8', errors='replace')
                logger.info("Loaded CSV with UTF-8 and replaced invalid characters")
        
        # Build ground truth mapping
        self.gt_mapping = {}
        for _, row in self.ground_truth.iterrows():
            question = self.normalize_question(row['Question'])
            answer = str(row['Answer'])
            self.gt_mapping[question] = answer
        
        logger.info(f"Loaded {len(self.ground_truth)} ground truth entries")
    
    def normalize_question(self, question: str) -> str:
        """Normalize question for matching"""
        # Remove punctuation and extra whitespace
        question = re.sub(r'[^\w\s]', ' ', question.lower())
        question = re.sub(r'\s+', ' ', question.strip())
        return question
    
    def find_ground_truth(self, query: str) -> Optional[str]:
        """Find ground truth answer for a query with improved matching"""
        query_normalized = self.normalize_question(query)
        
        # Try exact match first
        if query_normalized in self.gt_mapping:
            return self.gt_mapping[query_normalized]
        
        # Try fuzzy matching with high similarity threshold
        best_match = None
        best_score = 0.0
        
        for gt_query, gt_answer in self.gt_mapping.items():
            similarity = SequenceMatcher(None, query_normalized, gt_query).ratio()
            if similarity > best_score and similarity > 0.85:  # High threshold
                best_score = similarity
                best_match = gt_answer
        
        if best_match:
            logger.debug(f"Fuzzy matched query: {query} -> {best_match} (score: {best_score:.3f})")
        
        return best_match
    
    def validate_response(self, response: str) -> bool:
        """Validate that response is not corrupted"""
        if not response:
            return False
        
        # Check for truncation indicators
        if "1Human:" in response:
            return False
        
        # Check for extremely short responses
        if len(response.strip()) < 10:
            return False
        
        # Check for garbled text
        if len(re.findall(r'[^\w\s\-.,!?;:()"]', response)) > len(response) * 0.1:
            return False
        
        return True
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all queries with robust validation"""
        logger.info("Evaluating all queries...")
        
        all_retrieval_metrics = defaultdict(list)
        all_generation_metrics = defaultdict(list)
        latency_metrics = []
        
        matched_queries = 0
        valid_responses = 0
        total_queries = len(self.query_logs)
        
        individual_results = []
        
        for i, query_log in enumerate(self.query_logs):
            if i % 5 == 0:
                logger.info(f"Processing query {i+1}/{total_queries}")
            
            query = query_log['query']
            response = query_log['response']
            retrieved_docs = query_log.get('retrieved_documents', [])
            latencies = query_log.get('latencies', {})
            
            # Validate response
            if not self.validate_response(response):
                logger.warning(f"Invalid response detected for query: {query[:50]}...")
                continue
            
            valid_responses += 1
            
            # Collect latency metrics
            if latencies:
                total_latency = query_log.get('total_time', 0)
                retrieval_latency = latencies.get('document_retrieval', 0)
                generation_latency = latencies.get('response_generation', 0)
                
                latency_metrics.append({
                    'total_time': total_latency,
                    'retrieval_time': retrieval_latency,
                    'generation_time': generation_latency
                })
            
            # Find ground truth
            ground_truth = self.find_ground_truth(query)
            
            if ground_truth:
                matched_queries += 1
                
                # Evaluate retrieval
                retrieval_metrics = self.retrieval_evaluator.evaluate_query(query, retrieved_docs, ground_truth)
                for metric, value in retrieval_metrics.items():
                    all_retrieval_metrics[metric].append(value)
                
                # Evaluate generation
                generation_metrics = self.generation_evaluator.evaluate_generation(ground_truth, response, query)
                for metric, value in generation_metrics.items():
                    all_generation_metrics[metric].append(value)
                
                # Store individual result
                individual_results.append({
                    'query': query,
                    'ground_truth': ground_truth,
                    'response': response,
                    'exact_match': generation_metrics.get('exact_match', 0),
                    'numerical_accuracy': generation_metrics.get('numerical_accuracy', 0),
                    'contains_answer': generation_metrics.get('contains_answer', 0)
                })
        
        # Calculate average metrics
        avg_retrieval_metrics = {k: np.mean(v) for k, v in all_retrieval_metrics.items()}
        avg_generation_metrics = {k: np.mean(v) for k, v in all_generation_metrics.items()}
        
        # Calculate latency statistics
        latency_stats = {}
        if latency_metrics:
            for metric in ['total_time', 'retrieval_time', 'generation_time']:
                values = [l[metric] for l in latency_metrics if metric in l and l[metric] > 0]
                if values:
                    latency_stats[metric] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        results = {
            'summary': {
                'total_queries': total_queries,
                'valid_responses': valid_responses,
                'matched_queries': matched_queries,
                'match_rate': matched_queries / total_queries if total_queries > 0 else 0,
                'validation_rate': valid_responses / total_queries if total_queries > 0 else 0
            },
            'retrieval_metrics': avg_retrieval_metrics,
            'generation_metrics': avg_generation_metrics,
            'latency_stats': latency_stats,
            'individual_results': individual_results
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted results"""
        print("\n" + "="*80)
        print("ROBUST RAG SYSTEM EVALUATION RESULTS")
        print("="*80)
        
        # Summary
        summary = results['summary']
        print(f"\nSUMMARY:")
        print(f"Total queries: {summary['total_queries']}")
        print(f"Valid responses: {summary['valid_responses']}")
        print(f"Matched queries: {summary['matched_queries']}")
        print(f"Match rate: {summary['match_rate']:.2%}")
        print(f"Validation rate: {summary['validation_rate']:.2%}")
        
        # Retrieval metrics
        if results['retrieval_metrics']:
            print(f"\nRETRIEVAL METRICS:")
            retrieval_metrics = results['retrieval_metrics']
            for k in [1, 3, 5, 10]:
                if f'precision@{k}' in retrieval_metrics:
                    print(f"  Precision@{k}: {retrieval_metrics[f'precision@{k}']:.4f}")
                if f'hit_rate@{k}' in retrieval_metrics:
                    print(f"  Hit Rate@{k}: {retrieval_metrics[f'hit_rate@{k}']:.4f}")
            if 'mrr' in retrieval_metrics:
                print(f"  MRR: {retrieval_metrics['mrr']:.4f}")
        
        # Generation metrics
        if results['generation_metrics']:
            print(f"\nGENERATION METRICS:")
            gen_metrics = results['generation_metrics']
            print(f"  Exact Match: {gen_metrics.get('exact_match', 0):.4f}")
            print(f"  Numerical Accuracy: {gen_metrics.get('numerical_accuracy', 0):.4f}")
            print(f"  Contains Answer: {gen_metrics.get('contains_answer', 0):.4f}")
            print(f"  Fuzzy Match: {gen_metrics.get('fuzzy_match', 0):.4f}")
            print(f"  Semantic Similarity: {gen_metrics.get('semantic_similarity', 0):.4f}")
            print(f"  ROUGE-1 F1: {gen_metrics.get('rouge_1_f', 0):.4f}")
            print(f"  ROUGE-L F1: {gen_metrics.get('rouge_l_f', 0):.4f}")
        
        # Latency stats
        if results['latency_stats']:
            print(f"\nLATENCY STATISTICS (seconds):")
            for metric, stats in results['latency_stats'].items():
                print(f"  {metric.replace('_', ' ').title()}:")
                print(f"    Mean: {stats['mean']:.4f}")
                print(f"    Median: {stats['median']:.4f}")
        
        # Show some individual results
        if 'individual_results' in results and results['individual_results']:
            print(f"\nSAMPLE RESULTS:")
            for i, result in enumerate(results['individual_results'][:3]):
                print(f"  Query {i+1}: {result['query']}")
                print(f"    Ground Truth: {result['ground_truth']}")
                print(f"    Response: {result['response'][:100]}...")
                print(f"    Exact Match: {result['exact_match']:.2f}")
                print(f"    Numerical Accuracy: {result['numerical_accuracy']:.2f}")
                print(f"    Contains Answer: {result['contains_answer']:.2f}")
                print()


def main():
    """Main evaluation function"""
    import os
    
    # Paths
    query_logs_path = os.path.join(os.path.dirname(__file__), "query_logs.json")
    ground_truth_path = os.path.join(os.path.dirname(__file__), "evaluation_sample.csv")
    output_path = os.path.join(os.path.dirname(__file__), "evaluation_results_robust.json")
    
    # Check if files exist
    if not os.path.exists(query_logs_path):
        print(f"Error: Query logs file not found at {query_logs_path}")
        return
    
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        return
    
    # Create evaluator
    evaluator = RobustRAGEvaluator(query_logs_path, ground_truth_path)
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, output_path)
    
    print(f"\nEvaluation complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()

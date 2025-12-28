"""
LLM Evaluation Metrics for RoundtableAI.

This module provides comprehensive evaluation metrics using:
- RAGAs library (Faithfulness, Answer Relevancy, Context Precision/Recall)
- ROUGE scores for text quality
- Classification metrics (Accuracy, Precision, Recall, F1)

Prerequisites:
    pip install ragas rouge-score datasets

Usage:
    from evaluation.llm_metrics import LLMEvaluator

    evaluator = LLMEvaluator()
    results = evaluator.evaluate_recommendation(
        query="Should I invest in Maybank?",
        response="Based on the analysis...",
        context=["Financial data...", "News articles..."],
        predicted_label="BUY",
        ground_truth_label="BUY"
    )
"""
import warnings
warnings.filterwarnings('ignore')

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics

# RAGAs imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAs not installed. Install with: pip install ragas datasets")

# LangChain imports for LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    LANGCHAIN_GOOGLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics."""
    # RAGAs metrics
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None

    # ROUGE scores
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Additional metadata
    predicted_label: Optional[str] = None
    ground_truth_label: Optional[str] = None
    is_correct: Optional[bool] = None


class RAGAsEvaluator:
    """
    RAGAs evaluation metrics using the official RAGAs library.

    RAGAs (Retrieval-Augmented Generation Assessment) provides LLM-as-judge
    metrics for evaluating RAG systems.

    Metrics:
    - Faithfulness: How grounded is the response in the retrieved context
    - Answer Relevancy: How relevant is the response to the query
    - Context Precision: How precise is the retrieved context
    - Context Recall: How complete is the retrieved context
    - Answer Similarity: Semantic similarity to reference answer
    - Answer Correctness: Factual correctness of the answer
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize RAGAs evaluator with Google Gemini.

        Args:
            model_name: Gemini model to use for evaluation
        """
        self.model_name = model_name
        self._llm = None
        self._embeddings = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of LLM and embeddings."""
        if self._initialized:
            return

        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAs is not installed. Install with:\n"
                "pip install ragas datasets"
            )

        if not LANGCHAIN_GOOGLE_AVAILABLE:
            raise ImportError(
                "langchain-google-genai is not installed. Install with:\n"
                "pip install langchain-google-genai"
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file.\n"
                "Get your API key from: https://aistudio.google.com/apikey"
            )

        # Initialize Google Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.0,  # Deterministic for evaluation
            google_api_key=api_key,
        )
        self._llm = LangchainLLMWrapper(llm)

        # Initialize Google embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        self._embeddings = LangchainEmbeddingsWrapper(embeddings)

        self._initialized = True
        print(f"RAGAs initialized with {self.model_name}")

    def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response using RAGAs metrics.

        Args:
            query: The user's original question
            response: The generated response from the system
            contexts: List of retrieved context strings
            reference: Optional reference answer for similarity metrics
            ground_truth: Optional ground truth answer for correctness metrics

        Returns:
            Dictionary with RAGAs metric scores
        """
        self._initialize()

        # Prepare data in RAGAs format
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts],
        }

        # Add reference/ground_truth if provided
        if reference:
            data["reference"] = [reference]
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Select metrics based on available data
        metrics_to_use = [faithfulness, answer_relevancy]

        if ground_truth:
            metrics_to_use.extend([context_precision, context_recall])

        if reference:
            metrics_to_use.append(answer_similarity)

        if ground_truth and reference:
            metrics_to_use.append(answer_correctness)

        try:
            # Run RAGAs evaluation
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            # Convert to dictionary
            scores = results.to_pandas().iloc[0].to_dict()

            return {
                'faithfulness': scores.get('faithfulness'),
                'answer_relevancy': scores.get('answer_relevancy'),
                'context_precision': scores.get('context_precision'),
                'context_recall': scores.get('context_recall'),
                'answer_similarity': scores.get('answer_similarity'),
                'answer_correctness': scores.get('answer_correctness'),
            }

        except Exception as e:
            print(f"RAGAs evaluation error: {e}")
            return {
                'faithfulness': None,
                'answer_relevancy': None,
                'context_precision': None,
                'context_recall': None,
                'answer_similarity': None,
                'answer_correctness': None,
            }

    def evaluate_batch(
        self,
        queries: List[str],
        responses: List[str],
        contexts_list: List[List[str]],
        references: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """
        Evaluate multiple RAG responses in batch.

        Args:
            queries: List of user questions
            responses: List of generated responses
            contexts_list: List of context lists (one per query)
            references: Optional list of reference answers
            ground_truths: Optional list of ground truth answers

        Returns:
            Dictionary with lists of metric scores
        """
        self._initialize()

        # Prepare batch data
        data = {
            "question": queries,
            "answer": responses,
            "contexts": contexts_list,
        }

        if references:
            data["reference"] = references
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Select metrics
        metrics_to_use = [faithfulness, answer_relevancy]
        if ground_truths:
            metrics_to_use.extend([context_precision, context_recall])
        if references:
            metrics_to_use.append(answer_similarity)

        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            df = results.to_pandas()
            return df.to_dict(orient='list')

        except Exception as e:
            print(f"RAGAs batch evaluation error: {e}")
            return {}


class ROUGEEvaluator:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics.

    Used to evaluate the quality of generated text against reference text.
    """

    def __init__(self):
        """Initialize ROUGE evaluator."""
        self._rouge_scorer = None

    def _get_scorer(self):
        """Lazy load rouge scorer."""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True
                )
            except ImportError:
                print("rouge-score not installed. Install with: pip install rouge-score")
                return None
        return self._rouge_scorer

    def calculate_rouge(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores between generated and reference text.

        Args:
            generated: The generated text
            reference: The reference/ground truth text

        Returns:
            Dictionary with rouge-1, rouge-2, rouge-L F1 scores
        """
        scorer = self._get_scorer()

        if scorer is None or not generated or not reference:
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}

        scores = scorer.score(reference, generated)

        return {
            'rouge_1': scores['rouge1'].fmeasure,
            'rouge_2': scores['rouge2'].fmeasure,
            'rouge_l': scores['rougeL'].fmeasure
        }


class ClassificationEvaluator:
    """
    Classification metrics for recommendation evaluation.

    Evaluates predicted recommendations (BUY/HOLD/SELL) against ground truth.
    """

    LABELS = ['BUY', 'HOLD', 'SELL']

    def __init__(self):
        """Initialize classification evaluator."""
        self.predictions: List[str] = []
        self.ground_truths: List[str] = []

    def add_prediction(self, predicted: str, ground_truth: str):
        """
        Add a prediction-ground truth pair.

        Args:
            predicted: Predicted label (BUY/HOLD/SELL)
            ground_truth: Ground truth label (BUY/HOLD/SELL)
        """
        self.predictions.append(predicted.upper())
        self.ground_truths.append(ground_truth.upper())

    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self.predictions:
            return 0.0

        correct = sum(1 for p, g in zip(self.predictions, self.ground_truths) if p == g)
        return correct / len(self.predictions)

    def calculate_metrics_per_class(self) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, F1 for each class."""
        results = {}

        for label in self.LABELS:
            tp = sum(1 for p, g in zip(self.predictions, self.ground_truths) if p == label and g == label)
            fp = sum(1 for p, g in zip(self.predictions, self.ground_truths) if p == label and g != label)
            fn = sum(1 for p, g in zip(self.predictions, self.ground_truths) if p != label and g == label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(1 for g in self.ground_truths if g == label)
            }

        return results

    def calculate_macro_metrics(self) -> Dict[str, float]:
        """Calculate macro-averaged precision, recall, F1."""
        per_class = self.calculate_metrics_per_class()
        classes_with_support = [c for c, m in per_class.items() if m['support'] > 0]

        if not classes_with_support:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        macro_precision = statistics.mean(per_class[c]['precision'] for c in classes_with_support)
        macro_recall = statistics.mean(per_class[c]['recall'] for c in classes_with_support)
        macro_f1 = statistics.mean(per_class[c]['f1'] for c in classes_with_support)

        return {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }

    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix."""
        matrix = {actual: {pred: 0 for pred in self.LABELS} for actual in self.LABELS}

        for pred, actual in zip(self.predictions, self.ground_truths):
            if actual in matrix and pred in matrix[actual]:
                matrix[actual][pred] += 1

        return matrix

    def reset(self):
        """Reset all stored predictions."""
        self.predictions = []
        self.ground_truths = []


class LLMEvaluator:
    """
    Comprehensive LLM evaluator combining RAGAs, ROUGE, and Classification metrics.

    Usage:
        evaluator = LLMEvaluator()

        # Evaluate a single recommendation
        result = evaluator.evaluate_recommendation(
            query="Should I invest in Maybank?",
            response="Based on analysis, I recommend BUY...",
            context=["PE ratio: 12.5...", "Positive sentiment..."],
            predicted_label="BUY",
            ground_truth_label="BUY"
        )

        # Get aggregate results
        summary = evaluator.get_aggregate_results()
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the comprehensive evaluator.

        Args:
            model_name: Gemini model to use for RAGAs evaluation
        """
        self.ragas = RAGAsEvaluator(model_name=model_name) if RAGAS_AVAILABLE else None
        self.rouge = ROUGEEvaluator()
        self.classification = ClassificationEvaluator()
        self.results: List[EvaluationResult] = []

    def evaluate_recommendation(
        self,
        query: str,
        response: str,
        context: List[str] = None,
        predicted_label: str = None,
        ground_truth_label: str = None,
        reference_response: str = None
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation of a recommendation.

        Args:
            query: The user's original question
            response: The generated response from the system
            context: List of retrieved context (financial data, news, etc.)
            predicted_label: The predicted recommendation (BUY/HOLD/SELL)
            ground_truth_label: The ground truth recommendation
            reference_response: Optional reference response for ROUGE/similarity

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult()

        # RAGAs metrics (if context provided)
        if self.ragas and context:
            try:
                ragas_scores = self.ragas.evaluate(
                    query=query,
                    response=response,
                    contexts=context,
                    reference=reference_response,
                    ground_truth=reference_response,
                )
                result.faithfulness = ragas_scores.get('faithfulness')
                result.answer_relevancy = ragas_scores.get('answer_relevancy')
                result.context_precision = ragas_scores.get('context_precision')
                result.context_recall = ragas_scores.get('context_recall')
                result.answer_similarity = ragas_scores.get('answer_similarity')
                result.answer_correctness = ragas_scores.get('answer_correctness')
            except Exception as e:
                print(f"RAGAs evaluation failed: {e}")

        # ROUGE scores (if reference provided)
        if reference_response:
            rouge_scores = self.rouge.calculate_rouge(response, reference_response)
            result.rouge_1 = rouge_scores['rouge_1']
            result.rouge_2 = rouge_scores['rouge_2']
            result.rouge_l = rouge_scores['rouge_l']

        # Classification metrics
        if predicted_label and ground_truth_label:
            result.predicted_label = predicted_label.upper()
            result.ground_truth_label = ground_truth_label.upper()
            result.is_correct = result.predicted_label == result.ground_truth_label
            self.classification.add_prediction(predicted_label, ground_truth_label)

        self.results.append(result)
        return result

    def get_aggregate_results(self) -> Dict[str, Any]:
        """
        Get aggregate results across all evaluations.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self.results:
            return {}

        # Aggregate RAGAs metrics
        def safe_mean(values):
            valid = [v for v in values if v is not None]
            return statistics.mean(valid) if valid else None

        faithfulness_scores = [r.faithfulness for r in self.results]
        relevancy_scores = [r.answer_relevancy for r in self.results]
        precision_scores = [r.context_precision for r in self.results]
        recall_scores = [r.context_recall for r in self.results]
        similarity_scores = [r.answer_similarity for r in self.results]
        correctness_scores = [r.answer_correctness for r in self.results]

        # ROUGE scores
        rouge_1_scores = [r.rouge_1 for r in self.results]
        rouge_2_scores = [r.rouge_2 for r in self.results]
        rouge_l_scores = [r.rouge_l for r in self.results]

        # Classification metrics
        classification_metrics = self.classification.calculate_macro_metrics()
        accuracy = self.classification.calculate_accuracy()

        return {
            'total_evaluations': len(self.results),

            # RAGAs metrics
            'avg_faithfulness': safe_mean(faithfulness_scores),
            'avg_answer_relevancy': safe_mean(relevancy_scores),
            'avg_context_precision': safe_mean(precision_scores),
            'avg_context_recall': safe_mean(recall_scores),
            'avg_answer_similarity': safe_mean(similarity_scores),
            'avg_answer_correctness': safe_mean(correctness_scores),

            # ROUGE metrics
            'avg_rouge_1': safe_mean(rouge_1_scores),
            'avg_rouge_2': safe_mean(rouge_2_scores),
            'avg_rouge_l': safe_mean(rouge_l_scores),

            # Classification metrics
            'accuracy': accuracy if self.classification.predictions else None,
            'macro_precision': classification_metrics.get('precision'),
            'macro_recall': classification_metrics.get('recall'),
            'macro_f1': classification_metrics.get('f1'),

            # Detailed metrics
            'per_class_metrics': self.classification.calculate_metrics_per_class() if self.classification.predictions else None,
            'confusion_matrix': self.classification.get_confusion_matrix() if self.classification.predictions else None
        }

    def generate_report(self) -> str:
        """
        Generate a human-readable evaluation report.

        Returns:
            Formatted report string
        """
        agg = self.get_aggregate_results()

        if not agg:
            return "No evaluations performed yet."

        report = []
        report.append("=" * 70)
        report.append("LLM EVALUATION REPORT - RoundtableAI (RAGAs)")
        report.append("=" * 70)
        report.append(f"\nTotal Evaluations: {agg['total_evaluations']}")
        report.append("")

        # RAGAs Section
        report.append("RAGAs METRICS (LLM-as-Judge Evaluation)")
        report.append("-" * 50)
        if agg.get('avg_faithfulness') is not None:
            report.append(f"  Faithfulness:        {agg['avg_faithfulness']:.2%}")
        if agg.get('avg_answer_relevancy') is not None:
            report.append(f"  Answer Relevancy:    {agg['avg_answer_relevancy']:.2%}")
        if agg.get('avg_context_precision') is not None:
            report.append(f"  Context Precision:   {agg['avg_context_precision']:.2%}")
        if agg.get('avg_context_recall') is not None:
            report.append(f"  Context Recall:      {agg['avg_context_recall']:.2%}")
        if agg.get('avg_answer_similarity') is not None:
            report.append(f"  Answer Similarity:   {agg['avg_answer_similarity']:.2%}")
        if agg.get('avg_answer_correctness') is not None:
            report.append(f"  Answer Correctness:  {agg['avg_answer_correctness']:.2%}")
        report.append("")

        # ROUGE Section
        if agg.get('avg_rouge_1') is not None:
            report.append("ROUGE SCORES (Text Overlap)")
            report.append("-" * 50)
            report.append(f"  ROUGE-1 (Unigram):   {agg['avg_rouge_1']:.2%}")
            report.append(f"  ROUGE-2 (Bigram):    {agg['avg_rouge_2']:.2%}")
            report.append(f"  ROUGE-L (LCS):       {agg['avg_rouge_l']:.2%}")
            report.append("")

        # Classification Section
        if agg.get('accuracy') is not None:
            report.append("CLASSIFICATION METRICS (BUY/HOLD/SELL)")
            report.append("-" * 50)
            report.append(f"  Accuracy:            {agg['accuracy']:.2%}")
            report.append(f"  Macro Precision:     {agg['macro_precision']:.2%}")
            report.append(f"  Macro Recall:        {agg['macro_recall']:.2%}")
            report.append(f"  Macro F1 Score:      {agg['macro_f1']:.2%}")
            report.append("")

            # Per-class metrics
            report.append("  Per-Class Metrics:")
            per_class = agg.get('per_class_metrics', {})
            for label in ['BUY', 'HOLD', 'SELL']:
                if label in per_class:
                    m = per_class[label]
                    report.append(f"    {label:4s}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")
            report.append("")

            # Confusion Matrix
            report.append("  Confusion Matrix (rows=actual, cols=predicted):")
            cm = agg.get('confusion_matrix', {})
            report.append("         BUY  HOLD  SELL")
            for actual in ['BUY', 'HOLD', 'SELL']:
                row = f"  {actual:4s}"
                for pred in ['BUY', 'HOLD', 'SELL']:
                    row += f"  {cm.get(actual, {}).get(pred, 0):4d}"
                report.append(row)

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

    def reset(self):
        """Reset all stored results."""
        self.results = []
        self.classification.reset()

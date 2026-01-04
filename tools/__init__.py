"""
Agent tool modules for RoundtableAI multi-agent system.

This package provides specialized tools for:
- Fundamental analysis (fundamental_tools.py)
- Sentiment analysis (sentiment_tools.py)
- Valuation analysis (valuation_tools.py)
- Configuration (config.py)
"""
from .config import (
    set_reference_date,
    get_reference_date,
    get_default_date
)
from .fundamental_tools import (
    finance_report_pull,
    rag_analysis,
    prepare_financial_context,
    get_domain_expertise_guidance,
    analyze_cash_flow,
    analyze_operations,
    identify_concerns,
    assess_objectives,
    generate_overall_assessment,
    validate_overall_data_quality
)
from .sentiment_tools import (
    get_recent_articles,
    get_article_sentiment
)
from .valuation_tools import (
    get_stock_data,
    analyze_stock_metrics
)

__all__ = [
    # Config
    'set_reference_date',
    'get_reference_date',
    'get_default_date',
    # Fundamental tools
    'finance_report_pull',
    'rag_analysis',
    'prepare_financial_context',
    'get_domain_expertise_guidance',
    'analyze_cash_flow',
    'analyze_operations',
    'identify_concerns',
    'assess_objectives',
    'generate_overall_assessment',
    'validate_overall_data_quality',
    # Sentiment tools
    'get_recent_articles',
    'get_article_sentiment',
    # Valuation tools
    'get_stock_data',
    'analyze_stock_metrics',
]

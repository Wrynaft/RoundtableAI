"""
Fundamental analysis tools for evaluating company financial health.

This module provides tools for:
- Pulling financial reports from MongoDB
- Analyzing cash flow quality
- Evaluating operational efficiency
- Identifying financial risks and concerns
- Assessing strategic objectives and growth
- Generating overall investment recommendations
"""
from typing import List, Optional
from pymongo import MongoClient
from urllib.parse import quote_plus
from langchain.tools import tool


def validate_overall_data_quality(financial_data: dict) -> dict:
    """
    Validate overall data quality across all retrieved reports.

    Args:
        financial_data: Dictionary of financial data sections

    Returns:
        Dictionary containing validation results:
        - validation_passed: Whether data quality is acceptable
        - completeness_score: Percentage of successful data retrieval
        - issues: List of data quality issues
        - recommendations: List of recommendations for improvement
    """
    validation_results = {
        "validation_passed": True,
        "completeness_score": 0,
        "issues": [],
        "recommendations": []
    }

    try:
        total_sections = 0
        successful_sections = 0

        for section, data in financial_data.items():
            total_sections += 1
            if isinstance(data, dict) and "error" not in data:
                successful_sections += 1
            elif isinstance(data, list) and "error" in data:
                validation_results["issues"].append(f"Failed to retrieve {section} data.")

        # Calculate completeness score
        if total_sections > 0:
            validation_results["completeness_score"] = (successful_sections / total_sections) * 100

        # Determine if validation passed
        validation_results["validation_passed"] = validation_results["completeness_score"] >= 75

        # Add recommendations
        if validation_results["completeness_score"] < 50:
            validation_results["recommendations"].append(
                "Consider using alternative data sources due to low data completeness"
            )
        elif validation_results["completeness_score"] < 75:
            validation_results["recommendations"].append(
                "Some financial data missing - analysis may be limited"
            )

    except Exception as e:
        validation_results["validation_passed"] = False
        validation_results["issues"].append(f"Error during validation: {str(e)}")

    return validation_results


@tool
def finance_report_pull(ticker: str, report_types: List[str] = None, periods: int = 4) -> dict:
    """
    Pulls the latest financial reports for a given ticker symbol with validation.

    Retrieves financial statements and metrics from MongoDB including:
    - Income statements (quarterly and annual)
    - Balance sheets (quarterly and annual)
    - Cash flow statements (quarterly and annual)
    - Key financial metrics

    Args:
        ticker: Stock ticker symbol (e.g., "1155.KL")
        report_types: List of report types to retrieve. Options:
                     ["income_statement", "balance_sheet", "cash_flow", "financials"]
                     If None, retrieves all types
        periods: Number of periods to retrieve (default: 4)

    Returns:
        Dictionary containing:
        - success: Whether the operation was successful
        - ticker: Stock ticker symbol
        - company_name: Full company name
        - sector: Company sector
        - industry: Company industry
        - market_cap: Market capitalization
        - periods_requested: Number of periods requested
        - financial_data: Dictionary of financial statements and metrics
        - data_quality_assessment: Data quality validation results
    """
    # Connect to MongoDB
    username = quote_plus("Wrynaft")
    password = quote_plus("Ryan@120104")
    client = MongoClient(
        f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0"
    )
    db = client['roundtable_ai']
    col = db['fundamentals']

    if report_types is None:
        report_types = ["income_statement", "balance_sheet", "cash_flow", "financials"]

    # Retrieve full document
    full_doc = col.find_one(
        {"ticker": ticker},
        {
            "company_name": 1,
            "sector": 1,
            "industry": 1,
            "quarterly_income": 1,
            "annual_income": 1,
            "data_quality_income": 1,
            "quarterly_balance_sheet": 1,
            "annual_balance_sheet": 1,
            "data_quality_balance_sheet": 1,
            "quarterly_cashflow": 1,
            "annual_cashflow": 1,
            "data_quality_cashflow": 1,
            "metrics": 1,
            "_id": 0
        }
    )

    if not full_doc:
        return {
            "success": False,
            "error": f"No financial data found for ticker: {ticker}"
        }

    company_name = full_doc.get("company_name")
    sector = full_doc.get("sector")
    industry = full_doc.get("industry")

    financial_data = {}

    # Extract requested report types
    for report_type in report_types:
        if report_type == "income_statement":
            financial_data["income_statement"] = {
                "quarterly": full_doc.get("quarterly_income"),
                "annual": full_doc.get("annual_income"),
                "data_quality": full_doc.get("data_quality_income")
            }
        elif report_type == "balance_sheet":
            financial_data["balance_sheet"] = {
                "quarterly": full_doc.get("quarterly_balance_sheet"),
                "annual": full_doc.get("annual_balance_sheet"),
                "data_quality": full_doc.get("data_quality_balance_sheet")
            }
        elif report_type == "cash_flow":
            financial_data["cash_flow"] = {
                "quarterly": full_doc.get("quarterly_cashflow"),
                "annual": full_doc.get("annual_cashflow"),
                "data_quality": full_doc.get("data_quality_cashflow")
            }
        elif report_type == "financials":
            financial_data["key_metrics"] = full_doc.get("metrics")

    # Validate overall data quality
    overall_quality = validate_overall_data_quality(financial_data)

    result = {
        "success": True,
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "market_cap": full_doc.get("metrics", {}).get("market_cap"),
        "periods_requested": periods,
        "financial_data": financial_data,
        "data_quality_assessment": overall_quality
    }

    return result


def prepare_financial_context(financial_data: dict) -> dict:
    """
    Prepare structured financial context for analysis.

    Args:
        financial_data: Raw financial data from finance_report_pull

    Returns:
        Structured dictionary of financial context
    """
    context = {
        "company_profile": {
            "name": financial_data.get("company_name"),
            "sector": financial_data.get("sector"),
            "industry": financial_data.get("industry"),
            "market_cap": financial_data.get("market_cap")
        }
    }

    fd = financial_data.get("financial_data", {})

    if "income_statement" in fd and isinstance(fd["income_statement"], dict):
        context["income_statement"] = fd["income_statement"]

    if "balance_sheet" in fd and isinstance(fd["balance_sheet"], dict):
        context["balance_sheet"] = fd["balance_sheet"]

    if "cash_flow" in fd and isinstance(fd["cash_flow"], dict):
        context["cash_flow"] = fd["cash_flow"]

    if "key_metrics" in fd:
        context["key_metrics"] = fd["key_metrics"]

    return context


def get_domain_expertise_guidance(analysis_focus: str) -> dict:
    """
    Get domain expertise guidance for financial analysis.

    Args:
        analysis_focus: Type of analysis to perform. Options:
                       "cash_flow", "operations", "concerns", "objectives", "comprehensive"

    Returns:
        Dictionary containing guidance for the specified analysis focus
    """
    # Core financial analysis principles
    base_guidance = {
        "general_principles": [
            "Analyze trends over multiple periods",
            "Compare metrics to industry benchmarks",
            "Consider economic and sector context",
            "Identify key financial ratios and their implications",
            "Look for consistency in financial performance"
        ]
    }

    if analysis_focus == "cash_flow" or analysis_focus == "comprehensive":
        base_guidance["cash_flow_expertise"] = {
            "key_metrics": [
                "Operating Cash Flow",
                "Free Cash Flow",
                "Cash Flow from Investing",
                "Cash Flow from Financing",
                "Cash Conversion Cycle"
            ],
            "analysis_points": [
                "Evaluate cash generation quality and sustainability",
                "Assess working capital management efficiency",
                "Analyze capital allocation decisions",
                "Review cash flow predictability and seasonality",
                "Compare cash flow to net income (quality of earnings)"
            ]
        }

    if analysis_focus == "operations" or analysis_focus == "comprehensive":
        base_guidance["operations_expertise"] = {
            "key_metrics": [
                "Gross Margin",
                "Operating Margin",
                "EBITDA Margin",
                "Asset Turnover",
                "Inventory Turnover",
                "Return on Assets"
            ],
            "analysis_points": [
                "Assess operational efficiency trends",
                "Evaluate cost structure and margin stability",
                "Analyze revenue growth drivers",
                "Review asset utilization effectiveness",
                "Compare operational metrics to industry peers"
            ]
        }

    if analysis_focus == "concerns" or analysis_focus == "comprehensive":
        base_guidance["risk_assessment_expertise"] = {
            "financial_risks": [
                "Liquidity risk (current ratio, quick ratio)",
                "Leverage risk (debt-to-equity, interest coverage)",
                "Profitability deterioration",
                "Working capital management issues",
                "Cash flow sustainability concerns"
            ],
            "red_flags": [
                "Declining gross margins",
                "Increasing debt levels",
                "Deteriorating cash flow",
                "Growing accounts receivable relative to sales",
                "Frequent accounting changes or restatements"
            ]
        }

    if analysis_focus == "objectives" or analysis_focus == "comprehensive":
        base_guidance["strategic_assessment_expertise"] = {
            "growth_indicators": [
                "Revenue growth consistency",
                "Market share trends",
                "R&D investment levels",
                "Capital expenditure patterns",
                "Return on invested capital"
            ],
            "strategic_focus_areas": [
                "Evaluate management's stated strategic goals",
                "Assess progress on key performance indicators",
                "Review competitive positioning",
                "Analyze investment in future growth",
                "Consider ESG and sustainability initiatives"
            ]
        }

    return base_guidance


def find_cash_flow_key(data: dict, possible_keys: List[str]) -> Optional[str]:
    """
    Find the correct key for cash flow data in financial statements.

    Args:
        data: Dictionary of financial data
        possible_keys: List of possible key names to search for

    Returns:
        The found key name, or None if not found
    """
    for item_name, _ in data.items():
        for key in possible_keys:
            if key.lower() in item_name.lower():
                return item_name
    return None


def analyze_cash_flow(financial_context: dict, guidance: dict) -> dict:
    """
    Analyze cash flow using financial context and domain expertise guidance.

    Args:
        financial_context: Prepared financial context
        guidance: Domain expertise guidance

    Returns:
        Dictionary containing cash flow analysis results
    """
    analysis = {
        "cash_flow_quality": "Unknown",
        "key_insights": [],
        "strengths": [],
        "concerns": [],
        "recommendations": []
    }

    try:
        cash_flow_data = financial_context.get("cash_flow", {})
        key_metrics = financial_context.get("key_metrics", {})

        if cash_flow_data and isinstance(cash_flow_data, dict):
            # Analyze quarterly cash flow trends
            quarterly_data = cash_flow_data.get("quarterly", {}).get("data", {})

            if quarterly_data:
                # Extract operating cash flow trends
                operating_cf_key = find_cash_flow_key(
                    quarterly_data,
                    ["Operating Cash Flow", "Total Cash From Operating Activities"]
                )

                if operating_cf_key:
                    cf_values = []
                    periods = []

                    for period, values in quarterly_data.items():
                        if operating_cf_key in values and values[operating_cf_key] is not None:
                            cf_values.append(values[operating_cf_key])
                            periods.append(period)

                    if len(cf_values) >= 2:
                        # Analyze cash flow trend
                        recent_cf = cf_values[0] if cf_values else 0
                        prior_cf = cf_values[1] if len(cf_values) > 1 else 0

                        if recent_cf > 0:
                            analysis["cash_flow_quality"] = "Positive"
                            if recent_cf > prior_cf:
                                analysis["strengths"].append("Operating cash flow is improving")
                            else:
                                analysis["concerns"].append("Operating cash flow is declining")
                        else:
                            analysis["cash_flow_quality"] = "Negative"
                            analysis["concerns"].append("Negative operating cash flow")

                        analysis["key_insights"].append(
                            f"Most recent operating cash flow: ${recent_cf:,.0f}"
                        )

        # Add domain expertise insights
        cash_flow_guidance = guidance.get("cash_flow_expertise", {})
        for analysis_point in cash_flow_guidance.get("analysis_points", []):
            analysis["recommendations"].append(f"Consider: {analysis_point}")

    except Exception as e:
        analysis["concerns"].append(f"Error during cash flow analysis: {str(e)}")

    return analysis


def analyze_operations(financial_context: dict, guidance: dict) -> dict:
    """
    Analyze operations and profitability using financial context and domain expertise.

    Args:
        financial_context: Prepared financial context
        guidance: Domain expertise guidance

    Returns:
        Dictionary containing operations analysis results
    """
    analysis = {
        "operational_efficiency": "Unknown",
        "key_insights": [],
        "strengths": [],
        "concerns": [],
        "recommendations": []
    }

    try:
        key_metrics = financial_context.get("key_metrics", {})
        financial_health = key_metrics.get("financial_health", {})

        # Analyze profitability margins
        gross_margin = financial_health.get("gross_margins")
        operating_margin = financial_health.get("operating_margins")
        profit_margin = financial_health.get("profit_margins")

        if gross_margin is not None:
            analysis["key_insights"].append(f"Gross Margin: {gross_margin:.2%}")
            if gross_margin > 0.3:
                analysis["strengths"].append("Strong gross margin indicates good pricing power")
            elif gross_margin < 0.1:
                analysis["concerns"].append("Low gross margin indicates pricing pressure")

        if operating_margin is not None:
            analysis["key_insights"].append(f"Operating Margin: {operating_margin:.2%}")
            if operating_margin > 0.15:
                analysis["strengths"].append("Strong operating margin indicates efficient operations")
            elif operating_margin < 0.05:
                analysis["concerns"].append("Low operating margin indicates operational challenges")

        if profit_margin is not None:
            analysis["key_insights"].append(f"Profit Margin: {profit_margin:.2%}")
            if profit_margin > 0.1:
                analysis["strengths"].append("Healthy profit margin indicates overall profitability")
            elif profit_margin < 0.03:
                analysis["concerns"].append("Thin profit margin indicates limited profitability")

        # Assess overall operational efficiency
        if gross_margin and operating_margin and profit_margin:
            if all(m > 0.1 for m in [gross_margin, operating_margin, profit_margin]):
                analysis["operational_efficiency"] = "Strong"
            elif any(m < 0.0 for m in [gross_margin, operating_margin, profit_margin]):
                analysis["operational_efficiency"] = "Poor"
            else:
                analysis["operational_efficiency"] = "Moderate"

        # Add domain expertise insights
        ops_guidance = guidance.get("operations_expertise", {})
        for analysis_point in ops_guidance.get("analysis_points", []):
            analysis["recommendations"].append(f"Consider: {analysis_point}")

    except Exception as e:
        analysis["concerns"].append(f"Error during operations analysis: {str(e)}")

    return analysis


def identify_concerns(financial_context: dict, guidance: dict) -> dict:
    """
    Identify potential areas of concern using financial context and domain expertise.

    Args:
        financial_context: Prepared financial context
        guidance: Domain expertise guidance

    Returns:
        Dictionary containing risk assessment results
    """
    analysis = {
        "risk_level": "Unknown",
        "key_concerns": [],
        "financial_risks": [],
        "red_flags": [],
        "recommendations": []
    }

    try:
        key_metrics = financial_context.get("key_metrics", {})
        financial_health = key_metrics.get("financial_health", {})
        valuation = key_metrics.get("valuation", {})

        concern_count = 0

        # Liquidity concerns
        current_ratio = financial_health.get("current_ratio")
        if current_ratio is not None and current_ratio < 1.0:
            analysis["financial_risks"].append("Low current ratio indicates potential liquidity issues")
            concern_count += 1

        # Leverage concerns
        debt_to_equity = financial_health.get("debt_to_equity")
        if debt_to_equity is not None and debt_to_equity > 2.0:
            analysis["financial_risks"].append("High debt-to-equity ratio indicates high leverage risk")
            concern_count += 1

        # Profitability concerns
        roe = financial_health.get("return_on_equity")
        if roe is not None and roe < 0.05:
            analysis["financial_risks"].append("Low return on equity indicates poor profitability")
            concern_count += 1

        # Valuation concerns
        pe_ratio = valuation.get("pe_ratio")
        if pe_ratio is not None and pe_ratio > 50:
            analysis["key_concerns"].append("High P/E ratio may indicate overvaluation")
            concern_count += 1
        elif pe_ratio is not None and pe_ratio < 0:
            analysis["red_flags"].append("Negative P/E ratio indicates losses")
            concern_count += 2

        # Overall risk assessment
        if concern_count > 3:
            analysis["risk_level"] = "High"
        elif concern_count >= 1:
            analysis["risk_level"] = "Moderate"
        else:
            analysis["risk_level"] = "Low"

        # Add domain expertise insights
        risk_guidance = guidance.get("risk_assessment_expertise", {})
        for red_flag in risk_guidance.get("red_flags", []):
            analysis["recommendations"].append(f"Monitor for: {red_flag}")

    except Exception as e:
        analysis["key_concerns"].append(f"Error during concern identification: {str(e)}")

    return analysis


def assess_objectives(financial_context: dict, guidance: dict) -> dict:
    """
    Assess progress towards strategic objectives using financial context and guidance.

    Args:
        financial_context: Prepared financial context
        guidance: Domain expertise guidance

    Returns:
        Dictionary containing strategic assessment results
    """
    analysis = {
        "strategic_progress": "Unknown",
        "growth_indicators": [],
        "strategic_strengths": [],
        "areas_for_improvement": [],
        "recommendations": []
    }

    try:
        key_metrics = financial_context.get("key_metrics", {})
        growth = key_metrics.get("growth", {})

        revenue_growth = growth.get("revenue_growth")
        earnings_growth = growth.get("earnings_growth")

        if revenue_growth is not None:
            analysis["growth_indicators"].append(f"Revenue Growth: {revenue_growth:.2%}")
            if revenue_growth > 0.1:
                analysis["strategic_strengths"].append("Strong revenue growth indicates market expansion")
            elif revenue_growth < 0.0:
                analysis["areas_for_improvement"].append("Negative revenue growth indicates declining business")

        if earnings_growth is not None:
            analysis["growth_indicators"].append(f"Earnings Growth: {earnings_growth:.2%}")
            if earnings_growth > 0.15:
                analysis["strategic_strengths"].append("Strong earnings growth indicates operational leverage")
            elif earnings_growth < 0.0:
                analysis["areas_for_improvement"].append("Declining earnings indicate profitability challenges")

        # Overall strategic progress assessment
        if revenue_growth and earnings_growth:
            if revenue_growth > 0.05 and earnings_growth > 0.05:
                analysis["strategic_progress"] = "Strong"
            elif revenue_growth < 0.0 or earnings_growth < 0.0:
                analysis["strategic_progress"] = "Concerning"
            else:
                analysis["strategic_progress"] = "Moderate"

        # Add domain expertise insights
        strategic_guidance = guidance.get("strategic_assessment_expertise", {})
        for focus_area in strategic_guidance.get("strategic_focus_areas", []):
            analysis["recommendations"].append(f"Evaluate: {focus_area}")

    except Exception as e:
        analysis["areas_for_improvement"].append(f"Error during objectives assessment: {str(e)}")

    return analysis


def generate_overall_assessment(analysis_result: dict, financial_context: dict) -> dict:
    """
    Generate overall fundamental assessment and investment recommendation.

    Args:
        analysis_result: Combined analysis results from all analysis functions
        financial_context: Prepared financial context

    Returns:
        Dictionary containing overall assessment and investment recommendation
    """
    assessment = {
        "investment_recommendation": "HOLD",
        "confidence_level": "Medium",
        "key_strengths": [],
        "key_concerns": [],
        "fundamental_score": 50,  # Out of 100
        "summary": ""
    }

    try:
        # Aggregate strengths and concerns
        total_strengths = 0
        total_concerns = 0

        for analysis_type, analysis_data in analysis_result.items():
            if isinstance(analysis_data, dict):
                strengths = analysis_data.get("strengths", [])
                concerns = analysis_data.get("concerns", [])
                assessment["key_strengths"].extend(strengths)
                assessment["key_concerns"].extend(concerns)
                total_strengths += len(strengths)
                total_concerns += len(concerns)

        # Calculate fundamental score
        if total_strengths + total_concerns > 0:
            strength_ratio = total_strengths / (total_strengths + total_concerns)
            assessment["fundamental_score"] = int(strength_ratio * 100)

        # Generate investment recommendation
        if assessment["fundamental_score"] >= 70:
            assessment["investment_recommendation"] = "BUY"
            assessment["confidence_level"] = "High"
        elif assessment["fundamental_score"] >= 60:
            assessment["investment_recommendation"] = "BUY"
            assessment["confidence_level"] = "Medium"
        elif assessment["fundamental_score"] <= 30:
            assessment["investment_recommendation"] = "SELL"
            assessment["confidence_level"] = "High"
        elif assessment["fundamental_score"] <= 40:
            assessment["investment_recommendation"] = "SELL"
            assessment["confidence_level"] = "Medium"
        else:
            assessment["investment_recommendation"] = "HOLD"
            assessment["confidence_level"] = "Medium"

        # Generate summary
        company_name = financial_context.get("company_profile", {}).get("name", "Company")
        assessment["summary"] = (
            f"Fundamental analysis of {company_name} reveals a score of {assessment['fundamental_score']}/100. "
            f"Key strengths include: {', '.join(assessment['key_strengths'][:3]) if assessment['key_strengths'] else 'None identified'}. "
            f"Areas of concern include: {', '.join(assessment['key_concerns'][:3]) if assessment['key_concerns'] else 'None identified'}."
        )

    except Exception as e:
        assessment["summary"] = f"Error generating overall assessment: {str(e)}"

    return assessment


@tool
def rag_analysis(ticker: str, analysis_focus: str = "comprehensive", financial_data: dict = None) -> dict:
    """
    Performs comprehensive fundamental analysis using RAG (Retrieval-Augmented Generation).

    Uses domain expertise guidance to analyze financial reports and answer specific
    questions about cash flow, operations, areas of concern, and progress towards objectives.

    Args:
        ticker: Stock ticker symbol (e.g., "1155.KL")
        analysis_focus: Type of analysis to perform. Options:
                       "cash_flow", "operations", "concerns", "objectives", "comprehensive"
        financial_data: Financial data from finance_report_pull (if None, will be fetched)

    Returns:
        Dictionary containing:
        - success: Whether analysis was successful
        - ticker: Stock ticker symbol
        - company_name: Full company name
        - sector: Company sector
        - industry: Company industry
        - analysis_focus: Type of analysis performed
        - analysis_results: Detailed analysis results by category
        - overall_assessment: Investment recommendation and score
    """
    if financial_data is None:
        # If no financial data provided, fetch it
        financial_data = finance_report_pull.invoke({"ticker": ticker})
        if not financial_data.get("success"):
            return financial_data

    company_name = financial_data.get("company_name", ticker)
    sector = financial_data.get("sector", "Unknown")
    industry = financial_data.get("industry", "Unknown")

    financial_context = prepare_financial_context(financial_data)
    analysis_guidance = get_domain_expertise_guidance(analysis_focus)
    analysis_results = {}

    if analysis_focus == "comprehensive":
        # Perform all types of analysis
        analysis_results["cash_flow_analysis"] = analyze_cash_flow(financial_context, analysis_guidance)
        analysis_results["operations_analysis"] = analyze_operations(financial_context, analysis_guidance)
        analysis_results["concerns_analysis"] = identify_concerns(financial_context, analysis_guidance)
        analysis_results["objectives_analysis"] = assess_objectives(financial_context, analysis_guidance)
    else:
        # Perform specific analysis
        if analysis_focus == "cash_flow":
            analysis_results["cash_flow_analysis"] = analyze_cash_flow(financial_context, analysis_guidance)
        elif analysis_focus == "operations":
            analysis_results["operations_analysis"] = analyze_operations(financial_context, analysis_guidance)
        elif analysis_focus == "concerns":
            analysis_results["concerns_analysis"] = identify_concerns(financial_context, analysis_guidance)
        elif analysis_focus == "objectives":
            analysis_results["objectives_analysis"] = assess_objectives(financial_context, analysis_guidance)

    overall_assessment = generate_overall_assessment(analysis_results, financial_context)

    result = {
        "success": True,
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "analysis_focus": analysis_focus,
        "analysis_results": analysis_results,
        "overall_assessment": overall_assessment
    }

    return result
